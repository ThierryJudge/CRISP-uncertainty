from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, MutableMapping, Tuple

import numpy as np


@dataclass
class PatientResult:
    """Collection of data relevant to a patient (split across multiple views).

    Args:
        id: Patient's identifier (in format "patient0123").
        views: Mapping between each view available for the patient and the data associated with the view.
    """

    id: str
    views: MutableMapping[str, "ViewResult"] = field(default_factory=dict)


@dataclass
class ViewResult:
    """Collection of data for evaluating uncertainty for one View.

    Args:
        img: Image (N, C, H, W
        gt: GT (N, H, W)
        pred: Array prediction for the view (N, K, H, W)
        uncertainty_map: Array of uncertainty map prediction (N, H, W)
        frame_uncertainty: Array of uncertainty scores, one for each frame (N,)
        view_uncertainty: Uncertainty value for the entire frame (1,)
    """

    img: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    uncertainty_map: np.ndarray
    frame_uncertainties: np.ndarray
    view_uncertainty: np.ndarray
    voxelspacing: Tuple[Real, Real, Real]
    instants: Mapping[str, int]
