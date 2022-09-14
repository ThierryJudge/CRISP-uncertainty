from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor, nn

from vital.modules.layers import conv2d_activation, conv2d_activation_bn


class CamusUnet(nn.Module):
    """U-Net architecture fine-tuned for the CAMUS dataset.

    This is inspired by the original U-Net work (link in the refs) but was slightly modified to perform especially well
    on the CAMUS dataset. However, it is still a generic architecture applicable to any segmentation task.


    The input size, the number and types of layers are different. The number of feature maps per layer is also
    different. It provided better results than a standard U-Net when applied on the CAMUS dataset (link in the refs).

    References:
        - Paper that introduced the U-Net model: https://arxiv.org/abs/1505.04597
        - Paper that fine-tuned the U-Net model for the CAMUS dataset: https://ieeexplore.ieee.org/document/8649738
    """

    def __init__(self, in_channels: int, out_channels: int, init_channels: int, use_batchnorm: bool = True):
        """Initializes class instance.

        Args:
            in_channels: Number of channels of the input image to segment.
            out_channels: Number of channels of the segmentation to predict.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        self.encoder = _UnetEncoder(in_channels, init_channels, use_batchnorm=use_batchnorm)
        self.bottleneck = self._block(init_channels * 4, init_channels * 4, use_batchnorm=use_batchnorm)
        self.decoder = _UnetDecoder(out_channels, init_channels, use_batchnorm=use_batchnorm)

    @staticmethod
    def _block(in_channels: int, out_channels: int, use_batchnorm: bool = True) -> nn.Module:
        """Defines a convolutional block that maintains the spatial resolution of the feature maps.

        Args:
            in_channels: Number of input channels to the convolutional block.
            out_channels: Number of channels to output by the convolutional block.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.

        Returns:
            Convolutional block that maintains the spatial resolution of the feature maps.
        """
        if use_batchnorm:
            conv_block = conv2d_activation_bn
        else:
            conv_block = conv2d_activation
        return nn.Sequential(
            OrderedDict(
                [
                    ("conv_block1", conv_block(in_channels, out_channels)),
                    ("conv_block2", conv_block(out_channels, out_channels)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        x, *connected_features = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x, *connected_features)


class _UnetEncoder(nn.Module):
    """Module making up the encoder half of the U-Net model fine-tuned for the CAMUS dataset."""

    def __init__(self, in_channels: int, init_channels: int, use_batchnorm: bool = True):
        """Initializes class instance.

        Args:
            in_channels: Number of channels of the input image to segment.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.encoder1, self.pool1 = self._block(in_channels, init_channels)
        self.encoder2, self.pool2 = self._block(init_channels, init_channels)
        self.encoder3, self.pool3 = self._block(init_channels, init_channels * 2)
        self.encoder4, self.pool4 = self._block(init_channels * 2, init_channels * 4)
        self.encoder5, self.pool5 = self._block(init_channels * 4, init_channels * 4)

    def _block(self, in_channels: int, out_channels: int) -> Tuple[nn.Module, nn.Module]:
        """Defines a convolutional block that downsamples by a factor of 2 the spatial resolution of the feature maps.

        Args:
            in_channels: Number of input channels to the convolutional block.
            out_channels: Number of channels to output by the convolutional block.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.

        Returns:
            Convolutional block that downsamples by a factor of 2 the spatial resolution of the feature maps.
        """
        return (
            CamusUnet._block(in_channels, out_channels, use_batchnorm=self.use_batchnorm),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``init_channels`` * 4, H // 10, W // 10), Extracted features with the lowest spatial resolution, before
            they have passed through the bottleneck block.
        """
        features = []
        features.append(self.encoder1(x))
        features.append(self.encoder2(self.pool1(features[-1])))  # Downsample to 1/2 size
        features.append(self.encoder3(self.pool2(features[-1])))  # Downsample to 1/4 size
        features.append(self.encoder4(self.pool3(features[-1])))  # Downsample to 1/8 size
        features.append(self.encoder5(self.pool4(features[-1])))  # Downsample to 1/16 size
        x = self.pool5(features[-1])  # Downsample to 1/32 size
        return (x, *features)


class _UnetDecoder(nn.Module):
    """Module making up the decoder half of the U-Net model fine-tuned for the CAMUS dataset."""

    def __init__(self, out_channels: int, init_channels: int, use_batchnorm: bool = True):
        """Initializes class instance.

        Args:
            out_channels: Number of channels of the segmentation to predict.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.upsample5, self.decoder5 = self._block((init_channels * 4) * 2, init_channels * 4)
        self.upsample4, self.decoder4 = self._block((init_channels * 4) * 2, init_channels * 4)
        self.upsample3, self.decoder3 = self._block((init_channels * 4) + (init_channels * 2), init_channels * 2)
        self.upsample2, self.decoder2 = self._block((init_channels * 2) + init_channels, init_channels)
        self.upsample1, self.decoder1 = self._block(init_channels * 2, init_channels // 2)

        self.classifier = nn.Conv2d(init_channels // 2, out_channels, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> Tuple[nn.Module, nn.Module]:
        """Defines a convolutional block that upsamples by a factor of 2 the spatial resolution of the feature maps.

        Args:
            in_channels: Number of input channels to the convolutional block.
            out_channels: Number of channels to output by the convolutional block.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.

        Returns:
            Convolutional block that upsamples by a factor of 2 the spatial resolution of the feature maps.
        """
        return (
            nn.Upsample(scale_factor=2, mode="nearest"),
            CamusUnet._block(in_channels, out_channels, use_batchnorm=self.use_batchnorm),
        )

    def forward(self, x: Tensor, *connected_features: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``init_channels`` * 4, H // 10, W // 10), Extracted features with the lowest spatial resolution,
                before they have passed through the bottleneck block.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        features = self.upsample5(x)
        features = self.decoder5(torch.cat((features, connected_features[4]), dim=1))
        features = self.upsample4(features)
        features = self.decoder4(torch.cat((features, connected_features[3]), dim=1))
        features = self.upsample3(features)
        features = self.decoder3(torch.cat((features, connected_features[2]), dim=1))
        features = self.upsample2(features)
        features = self.decoder2(torch.cat((features, connected_features[1]), dim=1))
        features = self.upsample1(features)
        features = self.decoder1(torch.cat((features, connected_features[0]), dim=1))

        return self.classifier(features)
