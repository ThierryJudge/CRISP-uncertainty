from typing import Sequence, Tuple

import numpy as np
import PIL
import torch
from PIL import Image
from PIL.Image import NEAREST
from skimage.util import crop
from torch import Tensor


def resize_image(image: np.ndarray, size: Tuple[int, int], resample: PIL.Image = NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: Input image to process. Must be in a format supported by PIL.
        size: Width and height dimensions of the processed image to output.
        resample: Resampling filter to use.

    Returns:
        Input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image


def remove_labels(segmentation: np.ndarray, labels_to_remove: Sequence[int], fill_label: int = 0) -> np.ndarray:
    """Removes labels from the segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ([N], H, W, [1|C]), Segmentation map from which to remove labels.
        labels_to_remove: Labels to remove.
        fill_label: Label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ([N], H, W, [1]), Categorical segmentation map with the specified labels removed.
    """
    seg = segmentation.copy()
    if seg.max() == 1 and seg.shape[-1] > 1:  # If the segmentation map is in one-hot format
        for label_to_remove in labels_to_remove:
            seg[..., fill_label] += seg[..., label_to_remove]
        seg = np.delete(seg, labels_to_remove, axis=-1)
    else:  # the segmentation map is categorical
        seg[np.isin(seg, labels_to_remove)] = fill_label
    return seg


def segmentation_to_tensor(segmentation: np.ndarray, flip_channels: bool = False, dtype: str = "int64") -> Tensor:
    """Converts a segmentation map to a tensor, including reordering the dimensions.

    Args:
        segmentation: ([N], H, W, [C]), Segmentation map to convert to a tensor.
        flip_channels: If ``True``, assumes that the input is in `channels_last` mode and will automatically convert it
            to `channels_first` mode. If ``False``, leaves the ordering of dimensions untouched.
        dtype: Data type expected for the converted tensor, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        ([N], [C], H, W), Segmentation map converted to a tensor.

    Raises:
        ValueError: When reordering from `channels_last` to `channel_first`, the segmentation provided is neither 2D nor
            3D (only shapes supported when reordering channels).
    """
    if flip_channels:  # If there is a specific channel dimension
        if len(segmentation.shape) == 3:  # If it is a single segmentation
            dim_to_transpose = (2, 0, 1)
        elif len(segmentation.shape) == 4:  # If there is a batch dimension to keep first
            dim_to_transpose = (0, 3, 1, 2)
        else:
            raise ValueError(
                "Segmentation to convert to tensor is expected to be a single segmentation (2D), "
                "or a batch of segmentations (3D): \n"
                f"The segmentation to convert is {len(segmentation.shape)}D."
            )
        # Change format from `channel_last`, i.e. ([N], H, W, C), to `channel_first`, i.e. ([N], C, H, W)
        segmentation = segmentation.transpose(dim_to_transpose)
    return torch.from_numpy(segmentation.astype(dtype))


def centered_pad(image: np.ndarray, pad_size: Tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Pads the image, or batch of images, so that (H, W) match the requested `pad_size`.

    Args:
        image: ([N], H, W, C), Data to be padded.
        pad_size: (H, W) of the image after padding.
        pad_val: Value used for padding.

    Returns:
        ([N], H, W, C), Image, or batch of images, padded so that (H, W) match `pad_size`.
    """
    im_size = np.array(pad_size)

    if image.ndim == 4:
        to_pad = (im_size - image.shape[1:3]) // 2
        to_pad = ((0, 0), (to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))
    else:
        to_pad = (im_size - image.shape[:2]) // 2
        to_pad = ((to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))

    return np.pad(image, to_pad, mode="constant", constant_values=pad_val)


def centered_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """Crops the image, or batch of images, so that (H, W) match the requested `crop_size`.

    Args:
        image: ([N], H, W, C), Data to be cropped.
        crop_size: (H, W) of the image after the crop.

    Returns:
         ([N], H, W, C), Image, or batch of images, cropped so that (H, W) match `crop_size`.
    """
    if image.ndim == 4:
        to_crop = (np.array(image.shape[1:3]) - crop_size) // 2
        to_crop = ((0, 0), (to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    else:
        to_crop = (np.array(image.shape[:2]) - crop_size) // 2
        to_crop = ((to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    return crop(image, to_crop)


def centered_resize(image: np.ndarray, size: Tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Centers image around the requested `size`, either cropping or padding to match the target size.

    Args:
        image:  ([N], H, W, C), Data to be adapted to fit the target (H, W).
        size: Target (H, W) for the input image.
        pad_val: The value used for the padding.

    Returns:
        ([N], H, W, C), Image, or batch of images, adapted so that (H, W) match `size`.
    """
    if image.ndim == 4:
        height, width = image.shape[1:3]
    else:
        height, width = image.shape[:2]

    # Check the height to select if we crop of pad
    if size[0] - height < 0:
        image = centered_crop(image, (size[0], width))
    elif size[0] - height > 0:
        image = centered_pad(image, (size[0], width), pad_val)

    # Check if we crop or pad along the width dim of the image
    if size[1] - width < 0:
        image = centered_crop(image, size)
    elif size[1] - width > 0:
        image = centered_pad(image, size, pad_val)

    return image
