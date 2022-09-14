from typing import Tuple, Union

import torch.nn as nn
import vital.modules.segmentation.unet as vital_unet
from torch import Tensor


class UNet(vital_unet.UNet):
    """Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation.

    References:
    - Paper that introduced the U-Net model: https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        init_channels: int = 32,
        use_batchnorm: bool = True,
        bilinear: bool = False,
        dropout: float = 0.0,
        sigma_out: bool = False,
        conf_out: bool = False,
        confidnet: bool = False,
    ):
        """Initializes class instance.

        Args:
            input_shape: Shape of the input images.
            output_shape: Shape of the output segmentation map.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
            bilinear: Whether to use bilinear interpolation or transposed
                convolutions for upsampling.
            dropout: probability from dropout layers.
            sigma_out: When ``True`` second output is added to predict variance.
            conf_out: When ``True`` second output is added to predict confidence.
            confidnet: When ``True`` second output is added to predict confidnet output.
        """
        super().__init__(input_shape, output_shape, init_channels, use_batchnorm, bilinear, dropout)
        out_channels = output_shape[0]
        self.sigma_out = sigma_out
        self.conf_out = conf_out
        self.confidnet = confidnet

        assert not (sigma_out and conf_out)

        if sigma_out:
            self.layer11_sigma = vital_unet._Up(init_channels, init_channels // 2, 0, use_batchnorm, bilinear=bilinear)
            self.layer12_sigma = nn.Sequential(
                nn.Conv2d(init_channels // 2, out_channels, kernel_size=1), nn.Softplus()
            )

        if conf_out:
            self.layer11_conf = vital_unet._Up(init_channels, init_channels // 2, 0, use_batchnorm, bilinear=bilinear)
            self.layer12_conf = nn.Conv2d(init_channels // 2, 1, kernel_size=1)

        if confidnet:
            self.layer7_conf = vital_unet._Up(
                init_channels * 16, init_channels * 8, dropout, use_batchnorm, bilinear=bilinear
            )
            self.layer8_conf = vital_unet._Up(
                init_channels * 8, init_channels * 4, dropout, use_batchnorm, bilinear=bilinear
            )
            self.layer9_conf = vital_unet._Up(
                init_channels * 4, init_channels * 2, dropout, use_batchnorm, bilinear=bilinear
            )
            self.layer10_conf = vital_unet._Up(
                init_channels * 2, init_channels, dropout, use_batchnorm, bilinear=bilinear
            )
            self.layer11_conf = vital_unet._Up(init_channels, init_channels // 2, 0, use_batchnorm, bilinear=bilinear)

            self.layer12_conf = nn.Conv2d(init_channels // 2, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        out = self.layer7(x6, x5)
        out = self.layer8(out, x4)
        out = self.layer9(out, x3)
        out = self.layer10(out, x2)

        if self.sigma_out:
            logits = self.layer11(out, x1)
            sigma = self.layer11_sigma(out, x1)
            return self.layer12(logits), self.layer12_sigma(sigma)
        elif self.conf_out:
            logits = self.layer11(out, x1)
            sigma = self.layer11_conf(out, x1)
            return self.layer12(logits), self.layer12_conf(sigma)
        elif self.confidnet:
            out = self.layer11(out, x1)

            out_conf = self.layer7_conf(x6, x5)
            out_conf = self.layer8_conf(out_conf, x4)
            out_conf = self.layer9_conf(out_conf, x3)
            out_conf = self.layer10_conf(out_conf, x2)
            out_conf = self.layer11_conf(out_conf, x1)
            return self.layer12(out), self.layer12_conf(out_conf)
        else:
            out = self.layer11(out, x1)
            return self.layer12(out)


"""
This script can be run to visualize the network layers.
"""
if __name__ == "__main__":
    from torchsummary import summary

    model = UNet(input_shape=(1, 256, 256), output_shape=(4, 256, 256), sigma_out=True)

    summary(model, (1, 256, 256), device="cpu")
