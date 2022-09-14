import numpy as np
import torch
import torchvision.transforms.functional as F
from albumentations import ImageOnlyTransform
from skimage.draw import random_shapes
from skimage.filters import gaussian
from torch import Tensor

from vital.utils.image.transform import segmentation_to_tensor


class NormalizeSample(torch.nn.Module):
    """Normalizes a tensor w.r.t. to its mean and standard deviation.

    Args:
        inplace: Whether to make this operation in-place.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor) -> Tensor:
        """Normalizes input tensor.

        Args:
            tensor: Tensor to normalize.

        Returns:
            Normalized image.
        """
        return F.normalize(tensor, [float(tensor.mean())], [float(tensor.std())], self.inplace)


class SegmentationToTensor(torch.nn.Module):
    """Converts a segmentation map to a tensor."""

    def __call__(self, data: np.ndarray) -> Tensor:
        """Converts the segmentation map to a tensor.

        Args:
            segmentation: ([N], H, W), Segmentation map to convert to a tensor.

        Returns:
            ([N], H, W), Segmentation map converted to a tensor.
        """
        return segmentation_to_tensor(data)


class GrayscaleToRGB(torch.nn.Module):
    """Converts grayscale image to RGB image where r == g == b."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Converts grayscale image to RGB image where r == g == b.

        Args:
            img: (N, 1, ...), Grayscale image to convert to RGB.

        Returns:
            (N, 3, ...), RGB version of the original grayscale image, where r == g == b.
        """
        if img.shape[1] == 1:
            repeat_sizes = [1] * img.ndim
            repeat_sizes[1] = 3
            img = img.repeat(*repeat_sizes)
        else:
            raise ValueError(
                f"{self.__class__.__name__} only supports converting single channel grayscale images to RGB images "
                f"where r == g == b. The image data you provided consists of {img.shape[1]} channel images."
            )
        return img


class DiffusedNoise(ImageOnlyTransform):
    """Diffused Noise

    Reference:
    "

    https://github.com/raghavian/lungVAE/blob/9bfc6b940b762b378e11d2fea31e944a2eaf14a4/data/dataset.py#L51



    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            s: int = 50,
            min_shapes: int = 10,
            max_shapes: int = 20,
            always_apply=False,
            p=0.5,
    ):
        super(DiffusedNoise, self).__init__(always_apply, p)
        self.s = s
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.seed = 0

    def apply(self, img, **params):
        skMask = (random_shapes((img.shape[0], img.shape[1]), min_shapes=self.min_shapes, max_shapes=self.max_shapes,
                                min_size=self.s, allow_overlap=True,
                                multichannel=False, shape='circle',
                                random_seed=self.seed)[0] < 128).astype(float)
        self.seed += 1
        skMask = gaussian(skMask, sigma=self.s / 2)  # 0.8 is ~dense opacity
        img += (0.6 * skMask - 0.1)
        return img
