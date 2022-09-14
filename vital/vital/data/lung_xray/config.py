from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

import numpy as np

from vital.data.config import DataTag, Tags


class Label(DataTag):
    """Enumeration of tags related to the different anatomical structures segmented in the dataset.

    Attributes:
        BG: Label of the background.
        FG: Label of the forground (lung) class
    """

    BG = 0
    FG = 1


@dataclass(frozen=True)
class View:
    """Collection of tags related to the different views available for each patient.

    Args:
        front: Tag referring to the posteroanterior (pa) view
    """

    PA: str = "pa"


in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
