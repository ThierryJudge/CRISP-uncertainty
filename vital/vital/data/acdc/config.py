from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from vital.data.config import DataTag, Tags


class Label(DataTag):
    """Enumeration of tags related to the different anatomical structures segmented in the dataset.

    Attributes:
        BG: Label of the background.
        RV: Label of the right ventricle.
        MYO: Label of the myocardium.
        LV: Label of the left ventricle.
    """

    BG = 0
    RV = 1
    MYO = 2
    LV = 3


class Instant(DataTag):
    """Collection of tags related to noteworthy 3D volumes of 2D MRI slices.

    Args:
        ED: Tag referring to the end-diastolic volume.
        ES: Tag referring to the end-systolic volume.
        MID: Tag referring to a sample between ED and ES.
    """

    ED = "ED"
    ES = "ES"
    MID = "MID"


@dataclass(frozen=True)
class AcdcTags(Tags):
    """Class to gather the tags referring to the different types of data stored in the HDF5 datasets.

    Args:
        registered: name of the tag indicating whether the dataset was registered.
        voxel_spacing: name of the tag referring to metadata indicating the voxel size used in the output
        slice_index: name of the tag referring to the index of the slice
        proc_slices: Tag referring to metadata indicating which image were affected by the postprocessing.
    """

    registered: str = "register"
    voxel_spacing: str = "voxel_size"
    slice_index: str = "slice_index"
    proc_slices: str = "processed_slices"


image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the MRI image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
