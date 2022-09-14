from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

import numpy as np
from torch import Tensor


@dataclass
class PatientData:
    """Data structure that bundles data from the ACDC dataset for one patient.

    Args:
        id: patient's identifier (in format "patient0123").
        instants: Mapping between each instant available for the patient and the data associated with the instant.
    """

    id: str
    instants: MutableMapping[str, "InstantData"] = field(default_factory=dict)


@dataclass
class InstantData:
    """Data structure that bundles data from the ACDC dataset for one Instant (ED or ES).

    Args:
        img: Resized images, used as input when training models.
        gt: Resized groundtruths, used as target when training models.
        voxelspacing: Size of the segmentations' voxels along each (height, width, depth) dimension (in mm).
        registering: Parameters applied originally to register the images and groundtruths.
    """

    img: Tensor
    gt: Tensor
    voxelspacing: np.ndarray
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)
