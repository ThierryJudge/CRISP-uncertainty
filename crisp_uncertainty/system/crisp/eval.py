import math
import os
from pathlib import Path
from typing import Tuple

import comet_ml  # noqa
import hydra
import numpy as np
import torch
from pytorch_lightning import seed_everything
from scipy.stats import multivariate_normal
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from tqdm import tqdm
from vital.data.camus.data_struct import ViewData
from vital.data.config import Tags

from crisp_uncertainty.evaluation.data_struct import ViewResult
from crisp_uncertainty.evaluation.uncertainty.overlap import UncertaintyErrorOverlap
from crisp_uncertainty.system.crisp.crisp import CRISP
from crisp_uncertainty.system.uncertainty import UncertaintyEvaluationSystem
from matplotlib import pyplot as plt

from crisp_uncertainty.utils.numpy import prob_to_categorical


class EvalCRISP(UncertaintyEvaluationSystem, CRISP):
    module: nn.Module

    def __init__(
            self,
            module: nn.Module,
            module_ckpt: str = None,
            clip_ckpt: str = None,
            variance_factor: float = -1,
            num_samples: int = 25,
            samples_path: str = None,
            save_samples: Path = None,
            iterations: int = 1,
            decode: bool = False,
            *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            module: Module to train.
            clip_ckpt: Path to pre-trained crisp model.
            module_ckpt: Path to pre-trained segmentation model.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore='module')
        self.module = module

        checkpoint = torch.load(hydra.utils.to_absolute_path(self.hparams.clip_ckpt))
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        checkpoint = torch.load(hydra.utils.to_absolute_path(self.hparams.module_ckpt))
        self.load_state_dict(checkpoint["state_dict"], strict=False)

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        module_path = Path(self.hparams.module_ckpt).stem
        if int(module_path.rpartition('-')[-1]) == int(os.environ.get("PL_GLOBAL_SEED")):
            module_path = module_path.rpartition('-')[0]
        return f"CRISP{'*' if self.seg_channels > len(self.hparams.data_params.labels) else ''}" \
               f"_{module_path}_{self.hparams.num_samples}"

    def encode_set(self, dataloader):
        train_set_features = []
        train_set_segs = []
        for batch in tqdm(iter(dataloader)):
            seg = batch[Tags.gt].to(self.device)
            train_set_segs.append(seg.cpu())
            if self.hparams.data_params.out_shape[0] > 1:
                seg = to_onehot(seg, num_classes=self.hparams.data_params.out_shape[0]).float()
            else:
                seg = seg.unsqueeze(1).float()
            seg_mu = self.seg_encoder(seg)
            train_set_features.append(seg_mu.detach().cpu())
        return train_set_features, train_set_segs

    def on_test_epoch_start(self) -> None:
        print("Generate train features")

        datamodule = self.datamodule or self.trainer.datamodule

        self.to(self.device)

        if self.hparams.samples_path is None:
            datamodule.setup('fit')
            train_set_features, train_set_segs = self.encode_set(datamodule.train_dataloader())
            val_set_features, val_set_segs = self.encode_set(datamodule.val_dataloader())
            train_set_features.extend(val_set_features)
            train_set_segs.extend(val_set_segs)

            self.train_set_features = torch.cat(train_set_features)
            self.train_set_segs = torch.cat(train_set_segs)
            if self.hparams.save_samples:
                Path(self.hparams.save_samples).parent.mkdir(exist_ok=True)
                torch.save({'features': self.train_set_features,
                            'segmentations': self.train_set_segs}, self.hparams.save_samples)
                exit(0)
        else:
            samples = torch.load(self.hparams.samples_path)
            self.train_set_features = samples['features']
            self.train_set_segs = samples['segmentations']

        print("Latent features", self.train_set_features.shape)
        print("Latent segmentations", self.train_set_segs.shape)

        # TODO uncomment for augmented samples
        # if self.samples is not None:
        #     self.train_set_features = torch.cat([self.train_set_features, self.samples.to(self.device)])
        # self.train_set_segs = torch.cat(train_set_segs)
        # else:
        #     self.train_set_features = self.samples

        self.uncertainty_threshold = self.find_threshold(datamodule)
        self.log('best_uncertainty_threshold', self.uncertainty_threshold)

        seed_everything(0, workers=True)

    def compute_view_uncertainty(self, view: str, data: ViewData) -> ViewResult:
        # pred = self.module(data.img_proc.to(self.device))
        # pred = F.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)
        logits = [self.module(data.img_proc.to(self.device)) for _ in range(self.hparams.iterations)]
        if logits[0].shape[1] == 1:
            probs = [torch.sigmoid(logits[i]).detach() for i in range(self.hparams.iterations)]
            pred = torch.stack(probs, dim=-1).mean(-1)
        else:
            probs = [F.softmax(logits[i], dim=1).detach() for i in range(self.hparams.iterations)]
            pred = torch.stack(probs, dim=-1).mean(-1)

        frame_uncertainties = []
        uncertainty_maps = []
        for instant in range(pred.shape[0]):
            uncertainty, uncertainty_map = self.predict_uncertainty(data.img_proc[instant], pred[instant])
            frame_uncertainties.append(uncertainty)
            uncertainty_maps.append(uncertainty_map)

        return ViewResult(
            img=data.img_proc.cpu().numpy(),
            gt=data.gt_proc.cpu().numpy(),
            pred=pred.detach().cpu().numpy(),
            uncertainty_map=np.array(uncertainty_maps),
            frame_uncertainties=np.array(frame_uncertainties),
            view_uncertainty=np.mean(frame_uncertainties),
            voxelspacing=data.voxelspacing,
            instants=data.instants,
        )

    def predict_uncertainty(self, img: Tensor, pred: Tensor) -> Tuple[float, np.array]:
        img, pred = img.to(self.device), pred.to(self.device)
        if self.hparams.data_params.out_shape[0] > 1:
            pred = to_onehot(pred.argmax(0, keepdim=True), num_classes=self.hparams.data_params.out_shape[0])
        else:
            pred = pred.round().unsqueeze(1)

        # Get input image features
        img_mu = self.img_encoder(img[None])
        image_features = self.img_proj(img_mu)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get sample features
        samples = self.train_set_features.to(self.device).float()
        sample_features = self.seg_proj(samples)
        sample_features = sample_features / sample_features.norm(dim=-1, keepdim=True)
        sample_logits = sample_features @ image_features.t()

        # Get prediction features
        if self.hparams.variance_factor != -1:
            pred_mu = self.seg_encoder(pred.float())
            pred_features = self.seg_proj(pred_mu)
            pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
            pred_logits = pred_features @ image_features.t()

            cov = torch.abs(image_features - pred_features)
            cov = torch.sqrt(torch.sum(torch.square(image_features - pred_features))) * self.hparams.variance_factor
        else:
            sigma = torch.std(sample_features, dim=0)
            # cov = 1.06 * sigma * sample_features.shape[0]**(-1/5)
            q1, q3 = torch.quantile(sample_features, torch.tensor([0.25, 0.75]).to(self.device), dim=0, keepdim=True)
            cov = 0.9 * torch.minimum(sigma, (q3 - q1) / 1.34) * sample_features.shape[0] ** (-1 / 5)

        normal = multivariate_normal(mean=image_features.squeeze().cpu(), cov=cov.cpu().squeeze())
        gaussian_weights = normal.pdf(sample_features.squeeze().cpu()) / normal.pdf(image_features.squeeze().cpu())

        x_hat = torch.mean(sample_features, dim=0, keepdim=True)
        R_hat = torch.norm(x_hat)
        mu = x_hat / R_hat
        kappa = R_hat * (8 - R_hat**2) / 1 - R_hat**2
        # print(R_hat)
        # print(x_hat)
        # print(mu)
        # print(kappa)
        # print("Kappa: ", kappa.item())
        h0 = 7**(1/2) * (1 / kappa**(1/2)) * sample_features.shape[0]**(-1/5)
        # print("H0: ", h0)
        h0 = kappa**(-1/2) * (40 * torch.sqrt(torch.tensor(math.pi)) * sample_features.shape[0])**(-1/5)
        # print(h0)

        # print(h0, kappa)

        # print("H0: ", h0)
        # print("image w: ", torch.exp(h0 * image_features @ image_features.t()).squeeze())

        # weights = torch.exp(1 / h0 * image_features @ sample_features.t()).squeeze() / torch.exp(h0 * image_features @ image_features.t()).squeeze()
        # print(weights[:10])
        weights = torch.exp(1 / h0 * (image_features @ sample_features.t() - 1)).squeeze().cpu()

        # print(weights[:10])
        # weights = torch.exp(-h0 * torch.acos(mu @ sample_features.t())**2).squeeze()

        # print(weights.shape)


        # Get indices of samples higher than the prediction logits
        sorted_indices = torch.argsort(sample_logits, dim=0, descending=True)

        indices = sorted_indices[:self.hparams.num_samples]

        samples = samples[indices].squeeze()
        weights = weights[indices.cpu()]

        # print(gaussian_weights[indices.cpu()][:10])
        # print(weights[:10])
        #
        # from matplotlib import pyplot as plt
        # fig, axs = plt.subplots(1, 2, tight_layout=True)
        #
        #
        # print(gaussian_weights[gaussian_weights > 0.01].shape)
        # print(weights[weights > 0.01].shape)
        #
        # # We can set the number of bins with the *bins* keyword argument.
        # axs[0].hist(gaussian_weights[gaussian_weights > 0.01], bins=20)
        # axs[1].hist(weights[weights > 0.01].numpy(), bins=20)
        #
        # plt.show()

        decoded = self.seg_decoder(samples)
        # if self.hparams.decode:
        #     decoded = decoded.argmax(1) if decoded.shape[1] > 1 else torch.sigmoid(decoded).round()
        # else:
        decoded = self.train_set_segs[indices].squeeze()
        if self.seg_channels > 1:
            decoded = to_onehot(decoded, num_classes=self.seg_channels)

        # print(decoded.shape)
        # print(pred.shape)
        uncertainty_map = []
        aleatoric_map = []
        for i in range(decoded.shape[0]):

            # from matplotlib import pyplot as plt
            #
            # plt.figure()
            # plt.imshow(decoded[i].cpu().squeeze())
            #
            # plt.figure()
            # plt.imshow(pred.cpu().squeeze())
            # plt.show()

            weight = weights[i]
            # print(weight)
            diff = (~torch.eq(pred.squeeze().cpu(), decoded[i].cpu().squeeze())).float()
            uncertainty_map.append(diff[None] * weight)
            # aleatoric_map.append(decoded[i][None] * weight)

        uncertainty_map = torch.cat(uncertainty_map, dim=0)
        uncertainty_map = uncertainty_map.mean(0).squeeze()

        # aleatoric_map = torch.cat(aleatoric_map, dim=0)
        # mean_map = aleatoric_map.mean(0).squeeze()
        # std_map = aleatoric_map.std(0).squeeze()
        #
        # from matplotlib import pyplot as plt
        #
        # plt.figure()
        # plt.imshow(mean_map)
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(std_map)
        # plt.show()

        if self.seg_channels > 1:
            labels_values = [label.value for label in self.hparams.data_params.labels if label.value != 0]
            uncertainty_map = uncertainty_map[labels_values, ...]


        if uncertainty_map.ndim > 2:
            uncertainty_map = uncertainty_map.sum(0)

        uncertainty_map = uncertainty_map / uncertainty_map.max()
        # uncertainty_map = uncertainty_map.crisp(max=1)
        uncertainty_map = uncertainty_map.cpu().detach().numpy()

        # Compute frame uncertainty
        mask = pred.cpu().detach().numpy() != 0
        frame_uncertainty = (np.sum(uncertainty_map) / np.sum(mask))

        return frame_uncertainty, uncertainty_map

    def find_threshold(self, datamodule):
        if self.uncertainty_threshold == -1:
            print("Finding ideal threshold...")
            datamodule.setup('fit')  # Need to access Validation set
            val_dataloader = datamodule.val_dataloader()
            errors, uncertainties, error_sums = [], [], []

            for _, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Predicting on val set'):
                x, y = data[Tags.img], data[Tags.gt]
                pred = self.module(x.to(self.device))
                pred = F.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)
                for instant in range(pred.shape[0]):
                    _, unc = self.predict_uncertainty(x[instant], pred[instant])

                    err = ~np.equal(pred[instant].argmax(0).cpu().numpy(), y[instant].numpy())
                    errors.append(err)
                    uncertainties.append(unc[None])
                    error_sums.append(err.sum())

            errors, uncertainties, error_sums = np.concatenate(errors), np.concatenate(uncertainties), np.array(error_sums)
            print(errors.shape)
            print(uncertainties.shape)
            print(error_sums.shape)

            all_dices = []
            thresholds = np.arange(0.025, 1, 0.025)
            for thresh in tqdm(thresholds, desc='Finding ideal threshold'):
                dices = []
                for e, u in zip(errors, uncertainties):
                    dices.append(UncertaintyErrorOverlap.compute_overlap(e, u, thresh))
                all_dices.append(np.average(dices, weights=error_sums))

            return thresholds[np.argmax(all_dices)]
        else:
            return self.uncertainty_threshold
