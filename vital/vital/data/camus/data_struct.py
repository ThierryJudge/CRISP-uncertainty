from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, MutableMapping, Tuple

import numpy as np
from torch import Tensor


@dataclass
class PatientData:
    """Collection of data relevant to a patient (split across multiple views).

    Args:
        id: Patient's identifier (in format "patient0123").
        views: Mapping between each view available for the patient and the data associated with the view.
    """

    id: str
    views: MutableMapping[str, "ViewData"] = field(default_factory=dict)


@dataclass
class ViewData:
    """Collection of data relevant to a specific view sequence.

    Args:
        img_proc: Resized images, used as input when training models.
        gt_proc: Resized groundtruths, used as input when training models.
        gt: Unprocessed groundtruths, used as reference when evaluating models' scores.
        voxelspacing: Size of the segmentations' voxels along each (time, height, width) dimension (in mm).
        instants: Mapping between instant IDs and their frame index in the view.
        attrs: Attributes related to and/or computed on the image/groundtruth pair.
        registering: Parameters applied originally to register the images and groundtruths.
    """

    img_proc: Tensor
    gt_proc: Tensor
    gt: np.ndarray
    voxelspacing: Tuple[Real, Real, Real]
    instants: Mapping[str, int]
    attrs: Mapping[str, np.ndarray] = field(default_factory=dict)
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)
