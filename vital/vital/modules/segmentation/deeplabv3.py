from typing import Literal

import torchvision
from torch import Tensor, nn

from vital.data.transforms import GrayscaleToRGB


class DeepLabv3(nn.Module):
    """Wrapper around torchvision's implementation of the DeepLabv3 model that allows for single-channel inputs."""

    def __init__(
        self,
        backbone: Literal["resnet50", "resnet101"],
        num_classes: int,
        convert_grayscale_to_rgb: bool = False,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
    ):
        """Initializes class instance.

        Args:
            backbone: The network used by the DeepLabv3 architecture to compute the features for the model.
            num_classes: Number of output classes to segment.
            convert_grayscale_to_rgb: If ``True``, the forward pass will automatically convert single channel grayscale
                inputs to 3-channel RGB, where r == g == b, to fit with DeepLabv3's hardcoded 3 channel input layer.
                If ``False``, the input is assumed to already be 3 channel and is not transformed in any way.
            pretrained: Whether to use torchvision's pretrained weights for the DeepLabV3-specific modules.
            pretrained_backbone: Whether to use torchvision's pretrained weights for the backbone used by DeepLabV3,
                e.g. ResNet50.
        """
        super().__init__()
        self._convert_grayscale_to_rgb = convert_grayscale_to_rgb
        if self._convert_grayscale_to_rgb:
            self._grayscale_trans = GrayscaleToRGB()
        module_cls = torchvision.models.segmentation.__dict__[f"deeplabv3_{backbone}"]
        self._network = module_cls(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone, aux_loss=False, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x: (N, 1|3, H, W), Input image to segment.

        Returns:
            (N, ``num_classes``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        if self._convert_grayscale_to_rgb and x.shape[1] != 3:
            x = self._grayscale_trans(x)
        return self._network(x)["out"]
