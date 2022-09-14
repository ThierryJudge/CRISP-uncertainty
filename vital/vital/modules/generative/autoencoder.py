from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn

from vital.modules.generative.decoder import Decoder
from vital.modules.generative.encoder import Encoder
from vital.modules.layers import reparameterize


class Autoencoder(nn.Module):
    """Module making up a fully convolutional autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    reconstruction_tag: str = "x_hat"
    encoding_tag: str = "z"

    #:  Whether the encoder has a second head to output the ``logvar`` along with the default ``mu`` head
    output_distribution: bool = False

    def __init__(
        self,
        image_size: Tuple[int, int],
        channels: int,
        blocks: int,
        init_channels: int,
        latent_dim: int,
        input_latent_dim: int = 0,
        activation: str = "ELU",
        use_batchnorm: bool = True,
    ):
        """Initializes class instance.

        Args:
            image_size: Size of the output segmentation groundtruth for each axis.
            channels: Number of channels of the image to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            latent_dim: Number of dimensions in the latent space.
            input_latent_dim: Number of dimensions to add to the latent space prior to decoding. These are not predicted
                by the encoder, but come from auxiliary inputs to the network.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        self.encoder = Encoder(
            image_size=image_size,
            in_channels=channels,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            output_distribution=self.output_distribution,
        )
        self.decoder = Decoder(
            image_size=image_size,
            out_channels=channels,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim + input_latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: Tensor, subencodings: Sequence[Tensor] = None) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``channels``, H, W), Input to reconstruct.
            subencodings: (N, ?) tensors representing subspaces of the latent space, to be concatenated to the subspace
                predicted by the encoder to give the complete latent space vectors.
                When summed together, the second dimensions of these tensors should equal ``input_latent_dim``.

        Returns:
            Dict with values:
            - (N, ``channels``, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
            - (N, ``latent_dim``), Encoding of the input in the latent space.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z if subencodings is None else torch.cat((*subencodings, z), dim=1))
        return {self.reconstruction_tag: x_hat, self.encoding_tag: z}


class VariationalAutoencoder(Autoencoder):
    """Module making up a fully convolutional variational autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    distr_mean_tag: str = "mu"
    distr_logvar_tag: str = "logvar"

    output_distribution = True

    def forward(self, x: Tensor, subencodings: Sequence[Tensor] = None) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``channels``, H, W), Input to reconstruct.
            subencodings: (N, ?) tensors representing subspaces of the latent space, to be concatenated to the subspace
                predicted by the encoder to give the complete latent space vectors.
                When summed together, the second dimensions of these tensors should equal ``input_latent_dim``.

        Returns:
            Dict with values:
            - (N, ``channels``, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
            - (N, ``latent_dim``), Sampled encoding of the input in the latent space.
            - (N, ``latent_dim``), Mean of the predicted distribution of the input in the latent space, used to sample
              ``z``.
            - (N, ``latent_dim``), Log variance of the predicted distribution of the input in the latent space, used to
              sample ``z``.
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z if subencodings is None else torch.cat((*subencodings, z), dim=1))
        return {
            self.reconstruction_tag: x_hat,
            self.encoding_tag: z,
            self.distr_mean_tag: mu,
            self.distr_logvar_tag: logvar,
        }
