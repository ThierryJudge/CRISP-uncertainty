import random
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.utilities.data import to_onehot
from tqdm import tqdm
from vital.data.camus.config import CamusTags
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.systems.computation import TrainValComputationMixin

from crisp_uncertainty.system.crisp.crisp import CRISP


class TrainCRISP(CRISP, TrainValComputationMixin):

    def __init__(self, save_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('save_samples')

        if self.hparams.decode_img:
            self.img_reconstruction_loss = nn.MSELoss()

        if self.hparams.decode_seg:
            self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

    def trainval_step(self, batch: Any, batch_nb: int):
        img, seg = batch[Tags.img], batch[Tags.gt]
        if self.trainer.datamodule.data_params.out_shape[0] > 1:
            seg_onehot = to_onehot(seg, num_classes=self.trainer.datamodule.data_params.out_shape[0]).float()
        else:
            seg_onehot = seg.unsqueeze(1).float()

        logs = {}
        batch_size = img.shape[0]

        if self.hparams.output_distribution:
            img_mu, img_logvar = self.img_encoder(img)
            seg_mu, seg_logvar = self.seg_encoder(seg_onehot)
        else:
            img_mu = self.img_encoder(img)
            seg_mu = self.seg_encoder(seg_onehot)

        if self.hparams.interpolation_augmentation_samples > 0 and not self.is_val_step:
            augmented_samples = []
            for i in range(self.hparams.interpolation_augmentation_samples // 2):
                i1, i2 = random.randrange(len(img)), random.randrange(len(img))
                aug_seg1 = torch.lerp(seg_mu[i1], seg_mu[i2], random.uniform(-0.5, -1))
                aug_seg2 = torch.lerp(seg_mu[i1], seg_mu[i2], random.uniform(1.5, 2))
                augmented_samples.extend([aug_seg1[None], aug_seg2[None]])

            augmented_samples = torch.cat(augmented_samples, dim=0)
            augmentated_seg_mu = torch.cat([seg_mu, augmented_samples], dim=0)
        else:
            augmentated_seg_mu = seg_mu

        # Compute CLIP loss
        img_logits, seg_logits = self.clip_forward(img_mu, augmentated_seg_mu)

        labels = torch.arange(batch_size, device=self.device)
        loss_i = F.cross_entropy(img_logits, labels, reduction='none')
        loss_t = F.cross_entropy(seg_logits[:batch_size], labels, reduction='none')

        loss_i = loss_i.mean()
        loss_t = loss_t.mean()
        clip_loss = (loss_i + loss_t) / 2

        img_accuracy = img_logits.argmax(0)[:batch_size].eq(labels).float().mean()
        seg_accuracy = seg_logits.argmax(0).eq(labels).float().mean()

        loss = 0
        loss += self.hparams.clip_weight * clip_loss

        if self.hparams.linear_constraint_weight:
            regression_target = batch[self.hparams.linear_constraint_attr]
            regression = self.regression_module(seg_mu)
            regression_mse = F.mse_loss(regression, regression_target)
            loss += self.hparams.linear_constraint_weight * regression_mse
            logs.update({"regression_mse": regression_mse})

        # Compute VAE loss
        if self.hparams.decode_seg:
            if self.hparams.output_distribution:
                seg_z = self.reparameterize(seg_mu, seg_logvar)
                seg_kld = self.latent_space_metrics(seg_mu, seg_logvar)
            else:
                seg_kld = 0
                seg_z = seg_mu

            seg_recon = self.seg_decoder(seg_z)

            seg_metrics = self.seg_reconstruction_metrics(seg_recon, seg)

            seg_vae_loss = self.hparams.reconstruction_weight * seg_metrics['seg_recon_loss'] + \
                           self.hparams.kl_weight * seg_kld

            logs.update({'seg_vae_loss': seg_vae_loss, 'seg_kld': seg_kld})
            logs.update(seg_metrics)

            if self.is_val_step and batch_nb == 0:
                seg_recon = seg_recon.argmax(1) if seg_recon.shape[1] > 1 else torch.sigmoid(seg_recon).round()
                self.log_images(title='Sample (seg)', num_images=5,
                                axes_content={'Image': img.cpu().squeeze().numpy(),
                                              'GT': seg.cpu().squeeze().numpy(),
                                              'Pred': seg_recon.squeeze().detach().cpu().numpy()})

            loss += seg_vae_loss

        if self.hparams.attr_reg:
            attr_metrics = self._compute_latent_space_metrics(seg_mu, batch)
            attr_reg_sum = sum(attr_metrics[f"{attr}_attr_reg"] for attr in CamusTags.list_available_attrs(self.hparams.data_params.labels))
            loss += attr_reg_sum * 10
            logs.update({'attr_reg_loss': attr_reg_sum})

        if self.hparams.decode_img:
            if self.hparams.output_distribution:
                img_z = self.reparameterize(img_mu, img_logvar)
                img_kld = self.latent_space_metrics(img_mu, img_logvar)
            else:
                img_kld = 0
                img_z = img_mu

            img_recon = self.img_decoder(img_z)
            img_metrics = self.img_reconstruction_metrics(img_recon, img)

            img_vae_loss = self.hparams.reconstruction_weight * img_metrics['img_recon_loss'] + \
                           self.hparams.kl_weight * img_kld

            logs.update({'img_vae_loss': img_vae_loss, 'img_kld': img_kld, })
            logs.update(img_metrics)

            if self.is_val_step and batch_nb == 0:
                self.log_images(title='Sample (img)', num_images=5,
                                axes_content={'Image': img.cpu().squeeze().numpy(),
                                              'Pred': img_recon.squeeze().detach().cpu().numpy()})

            loss += img_vae_loss

        logs.update({
            'loss': loss,
            'clip_loss': clip_loss,
            'img_accuracy': img_accuracy,
            'seg_accuracy': seg_accuracy,
        })

        return logs

    def clip_forward(self, image_features, seg_features):
        image_features = self.img_proj(image_features)
        seg_features = self.seg_proj(seg_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        seg_features = seg_features / seg_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ seg_features.t()
        logits_per_seg = logit_scale * seg_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_seg

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def seg_reconstruction_metrics(self, recon_x: torch.Tensor, x: torch.Tensor):
        # Segmentation accuracy metrics
        if recon_x.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(recon_x.squeeze(), x.type_as(recon_x))
        else:
            ce = F.cross_entropy(recon_x, x)

        dice_values = self._dice(recon_x, x)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.cross_entropy_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))
        return {"seg_recon_loss": loss, "seg_ce": ce, "dice": mean_dice, **dices}

    def img_reconstruction_metrics(self, recon_x: torch.Tensor, x: torch.Tensor):
        return {"img_recon_loss": self.img_reconstruction_loss(recon_x, x)}

    def latent_space_metrics(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

    def _compute_latent_space_metrics(self, mu, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Computes metrics on the input's encoding in the latent space.
        Adds the attribute regularization term to the loss already computed by the parent's implementation.
        Args:
            out: Output of a forward pass with the autoencoder network.
            batch: Content of the batch of data returned by the dataloader.
        References:
            - Computation of the attribute regularization term inspired by the original paper's implementation:
              https://github.com/ashispati/ar-vae/blob/master/utils/trainer.py#L378-L403
        Returns:
            Metrics useful for computing the loss and tracking the system's training progress:
                - metrics computed by ``super()._compute_latent_space_metrics``
                - attribute regularization term for each attribute (under the "{attr}_attr_reg" label format)
        """
        attr_metrics = {}
        for attr_idx, attr in enumerate(CamusTags.list_available_attrs(self.hparams.data_params.labels)):
            # Extract dimension to regularize and target for the current attribute
            latent_code = mu[:, attr_idx].unsqueeze(1)
            attribute = batch[attr]

            # Compute latent distance matrix
            latent_code = latent_code.repeat(1, latent_code.shape[0])
            lc_dist_mat = latent_code - latent_code.transpose(1, 0)

            # Compute attribute distance matrix
            attribute = attribute.repeat(1, attribute.shape[0])
            attribute_dist_mat = attribute - attribute.transpose(1, 0)

            # Compute regularization loss
            # lc_tanh = torch.tanh(lc_dist_mat * self.hparams.delta)
            lc_tanh = torch.tanh(lc_dist_mat * 1)
            attribute_sign = torch.sign(attribute_dist_mat)
            attr_metrics[f"{attr}_attr_reg"] = F.l1_loss(lc_tanh, attribute_sign)

        return attr_metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

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

    def on_fit_end(self) -> None:
        if self.hparams.save_samples:

            print("Generate train features")

            datamodule = self.trainer.datamodule

            train_set_features, train_set_segs = self.encode_set(datamodule.train_dataloader())
            val_set_features, val_set_segs = self.encode_set(datamodule.val_dataloader())
            train_set_features.extend(val_set_features)
            train_set_segs.extend(val_set_segs)

            self.train_set_features = torch.cat(train_set_features)
            self.train_set_segs = torch.cat(train_set_segs)
            Path(self.hparams.save_samples).parent.mkdir(exist_ok=True)
            torch.save({'features': self.train_set_features,
                        'segmentations': self.train_set_segs}, self.hparams.save_samples)
