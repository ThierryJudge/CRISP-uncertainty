from typing import Any, Dict

import numpy as np

from vital.data.config import DataTag


class Label(DataTag):
    """Enumeration of tags related to the different anatomical structures segmented in the dataset.

    Attributes:
        BG: Label of the background.
        MYO: Label of the myocardium, bounded by the encocardium and epicardium.
    """

    BG = 0
    MYO = 1


image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the MRI image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
