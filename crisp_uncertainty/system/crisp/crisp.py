import itertools
from typing import Dict, List, Optional

import comet_ml  # noqa
import torch
import torch.nn as nn
from pytorch_lightning.core.memory import ModelSummary
from torchmetrics.utilities.data import to_onehot
from tqdm import tqdm
from vital.data.config import Tags
from vital.data.data_module import VitalDataModule
from vital.modules.generative.decoder import Decoder
from vital.modules.generative.encoder import Encoder
from vital.systems.system import SystemComputationMixin
import numpy as np


class CRISP(SystemComputationMixin):
    def __init__(
            self,
            img_latent_size=64,
            seg_latent_size=16,
            latent_size=16,
            img_blocks=6,
            seg_blocks=4,
            init_channels=48,
            decode_img: bool = False,
            decode_seg: bool = False,
            cross_entropy_weight: float = 0.1,
            dice_weight: float = 1,
            clip_weight: float = 1,
            reconstruction_weight: float = 2,
            kl_weight: float = 1,
            output_distribution: bool = False,
            linear_constraint_weight: float = 0,
            linear_constraint_attr: str = None,
            interpolation_augmentation_samples: int = 0,
            attr_reg: bool = False,
            seg_channels: int = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        input_shape = self.hparams.data_params.in_shape[1:3]
        img_channels = self.hparams.data_params.in_shape[0]
        self.seg_channels = seg_channels or self.hparams.data_params.out_shape[0]


        self.img_encoder = Encoder(input_shape, img_channels, self.hparams.img_blocks, self.hparams.init_channels,
                                   self.hparams.img_latent_size, output_distribution=self.hparams.output_distribution,
                                   use_batchnorm=True)
        self.seg_encoder = Encoder(input_shape, self.seg_channels, self.hparams.seg_blocks, self.hparams.init_channels,
                                   self.hparams.seg_latent_size, output_distribution=self.hparams.output_distribution,
                                   use_batchnorm=True)
        if self.hparams.decode_img:
            self.img_decoder = Decoder(input_shape, img_channels, self.hparams.img_blocks, self.hparams.init_channels,
                                       self.hparams.img_latent_size, use_batchnorm=True)
        if self.hparams.decode_seg:
            self.seg_decoder = Decoder(input_shape, self.seg_channels, self.hparams.seg_blocks, self.hparams.init_channels,
                                       self.hparams.seg_latent_size, use_batchnorm=True)

        self.img_proj = nn.Linear(self.hparams.img_latent_size, self.hparams.latent_size)
        self.seg_proj = nn.Linear(self.hparams.seg_latent_size, self.hparams.latent_size)

        self.logit_scale = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.register_parameter('logit_scale', self.logit_scale)

        if self.hparams.linear_constraint_weight:
            if self.hparams.linear_constraint_attr is None:
                raise ValueError(
                    "You requested a linearity constraint on the latent space by setting `linear_constraint_weight>0`, "
                    "but provided no attribute over which to compute this linearity constraint. Either provide an "
                    "attribute on which to compute the linearity constraint (through `--linear_constraint_attr` "
                    "parameter), or abandon the linearity constraint (set `--linear_constraint_weight` to None or 0)."
                )
            # TODO add regression module for img latent space.
            self.regression_module = nn.Linear(self.hparams.seg_latent_size, 1)

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        # No predefined forward method.
        pass

    def encode_dataset(self, datamodule: VitalDataModule, progress_bar: bool = False):
        """Encodes masks from the train and val sets of a dataset in the latent space learned by an CLIP model.

        Args:
            system: CLIP system used to encode masks in a latent space.
            datamodule: Abstraction of the dataset to encode, allowing to access both the training and validation sets.
            progress_bar: Whether to display a progress bar for the encoding of the samples.

        Returns:
            Array of training and validation samples encoded in the latent space.
        """
        # Setup the datamodule used to get the data points to encode in the latent space
        datamodule.setup(stage="fit")
        train_dataloader, val_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader()
        data = itertools.chain(train_dataloader, val_dataloader)
        num_batches = len(train_dataloader) + len(val_dataloader)

        if progress_bar:
            data = tqdm(data, desc="Encoding groundtruths", total=num_batches, unit="batch")

        # Encode training and validation groundtruths in the latent space
        dataset_samples = []
        with torch.no_grad():
            for batch in data:
                seg = batch[Tags.gt].to(self.device)
                if datamodule.data_params.out_shape[0] > 1:
                    seg = to_onehot(seg, num_classes=datamodule.data_params.out_shape[0]).float()
                else:
                    seg = seg.unsqueeze(1).float()
                dataset_samples.append(self.seg_encoder(seg).cpu())

        dataset_samples = torch.cat(dataset_samples).numpy()

        return dataset_samples

    def encode_dataloader(self, dataloader, output_shape):
        dataset_samples = []
        with torch.no_grad():
            for batch in dataloader:
                seg = batch[Tags.gt].to(self.device)
                if output_shape[0] > 1:
                    seg = to_onehot(seg, num_classes=output_shape[0]).float()
                else:
                    seg = seg.unsqueeze(1).float()
                dataset_samples.append(self.seg_encoder(seg).cpu())

        dataset_samples = torch.cat(dataset_samples).numpy()
        return dataset_samples


    def decode(self, encoding: np.ndarray) -> np.ndarray:
        """Decodes a sample, or batch of samples, from the latent space to the output space.

        Args:
            system: Autoencoder system with generative capabilities used to decode the encoded samples.
            encoding: Sample, or batch of samples, from the latent space to decode.

        Returns:
            Decoded sample, or batch of samples.
        """
        encoding = encoding.astype(np.float32)  # Ensure the encoding is in single-precision float dtype
        if len(encoding.shape) == 1:
            # If the input isn't a batch of value, add the batch dimension
            encoding = encoding[None, :]
        encoding_tensor = torch.from_numpy(encoding)
        decoded_sample = self.seg_decoder(encoding_tensor)
        decoded_sample = decoded_sample.argmax(1) if decoded_sample.shape[1] > 1 else torch.sigmoid(decoded_sample).round()

        return decoded_sample.squeeze().cpu().detach().numpy()



