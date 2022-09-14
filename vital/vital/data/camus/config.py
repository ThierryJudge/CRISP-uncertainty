from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

import numpy as np

from vital.data.config import DataTag, Tags


class Label(DataTag):
    """Enumeration of tags related to the different anatomical structures segmented in the dataset.

    Attributes:
        BG: Label of the background.
        LV: Label of the left ventricle, bounded by the endocardium.
        MYO: Label of the myocardium, bounded by the encocardium and epicardium.
        ATRIUM: Label of the left atrium.
    """

    BG = 0
    LV = 1
    MYO = 2
    ATRIUM = 3


@dataclass(frozen=True)
class View:
    """Collection of tags related to the different views available for each patient.

    Args:
        A2C: Tag referring to the apical two-chamber view.
        A4C: Tag referring to the apical four-chamber view.
    """

    A2C: str = "2CH"
    A4C: str = "4CH"


@dataclass(frozen=True)
class Instant:
    """Collection of tags related to noteworthy instants in ultrasound sequences.

    Args:
        ED: Tag referring to the end-diastolic instant.
        ES: Tag referring to the end-systolic instant.
    """

    @classmethod
    def from_sequence_type(cls, sequence_type: Literal["half_cycle", "full_cycle"]) -> "Instant":
        """Detects the specialized version of the `Instant` collection that fits the requested sequence type.

        Args:
            sequence_type: Flag that indicates the kind of sequences for which to provide the important instants.

        Returns:
            A specialized version of the `Instant` collection that fits the requested sequence type.
        """
        return globals()[f"{sequence_type.title().replace('_', '')}Instant"]()

    ED: str = "ED"
    ES: str = "ES"


@dataclass(frozen=True)
class HalfCycleInstant(Instant):
    """Collection of tags related to noteworthy instants in half-cycle ultrasound sequences."""

    pass


@dataclass(frozen=True)
class FullCycleInstant(Instant):
    """Collection of tags related to noteworthy instants in full-cycle ultrasound sequences.

    Args:
        ED_E: Tag referring to the end-diastolic instant marking the end of the cycle.
    """

    ED_E: str = "ED_E"


@dataclass(frozen=True)
class CamusTags(Tags):
    """Class to gather the tags referring to CAMUS specific data, from both the training and result datasets.

    Args:
        registered: Tag indicating whether the dataset was registered.
        full_sequence: Tag indicating whether the dataset contains complete sequence between ED and ES for each view.
        instants: Tag indicating the clinically important instants available in the sequence.
        img_proc: Tag referring to resized images, used as input when training models.
        gt_proc: Tag referring to resized groundtruths used as reference when training models.
        info: Tag referring to images' metadata.
        voxelspacing: Tag referring to voxels' size along each (time, height, width) dimension (in mm).
        proc_instants: Tag referring to metadata indicating which image where affected by the postprocessing.
        frame_pos: Tag referring to the frame normalized index in the sequence (normalized so that ED=0 and ES=1).
        lv_area: Tag referring to the number of pixels, in the groundtruths, associated to the left ventricle (LV).
        lv_base_width: Tag referring to the width of the LV's base, in the groundtruths.
        lv_length: Tag referring to the distance between the LV's base and apex, in the groundtruths.
        lv_orientation: Tag referring to the angle between the LV's main axis and the vertical.
        myo_area: Tag referring to the number of pixels, in the groundtruths, associated to the myocardium (MYO).
        epi_center_x: Tag referring to the x-coordinate of the epicardium's center of mass.
        epi_center_y: Tag referring to the y-coordinate of the epicardium's center of mass.
        atrium_area: Tag referring to the number of pixels, in the groundtruths, associated to the left atrium.
        seg_attrs: Collection of tags for attributes related to the segmentation sequences.
    """

    registered: str = "register"
    full_sequence: str = "sequence"
    instants: str = "instants"

    img_proc: str = "img_proc"
    gt_proc: str = "gt_proc"
    info: str = "info"
    voxelspacing: str = "voxelspacing"
    proc_instants: str = "processed_instants"

    frame_pos: str = "frame_pos"
    lv_area: str = "lv_area"
    lv_base_width: str = "lv_base_width"
    lv_length: str = "lv_length"
    lv_orientation: str = "lv_orientation"
    myo_area: str = "myo_area"
    epi_center_x: str = "epi_center_x"
    epi_center_y: str = "epi_center_y"
    atrium_area: str = "atrium_area"
    seg_attrs: Sequence[str] = (
        lv_area,
        lv_base_width,
        lv_length,
        lv_orientation,
        myo_area,
        epi_center_x,
        epi_center_y,
        atrium_area,
    )

    @classmethod
    def list_available_attrs(cls, labels: Sequence[Label]) -> List[str]:
        """Lists attributes that are available for a segmentation, given the labels provided in the segmentation.

        Args:
            labels: Labels provided in the segmentation, that determine what attributes can be extracted from the
                segmentation.

        Returns:
            Attributes available for the segmentation.
        """
        attrs = []
        if Label.LV in labels:
            attrs.extend([cls.lv_area, cls.lv_orientation])
        if Label.MYO in labels:
            attrs.append(cls.myo_area)
        if Label.LV in labels and Label.MYO in labels:
            attrs.extend([cls.lv_base_width, cls.lv_length, cls.epi_center_x, cls.epi_center_y])
        if Label.ATRIUM in labels:
            attrs.append(cls.atrium_area)
        return sorted(attrs, key=cls.seg_attrs.index)


in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
