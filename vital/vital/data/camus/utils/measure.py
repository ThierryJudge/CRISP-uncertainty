import logging
from typing import Tuple

import numpy as np
from scipy import ndimage
from skimage import morphology

from vital.data.camus.config import Label
from vital.data.config import SemanticStructureId
from vital.utils.decorators import auto_cast_data, batch_function
from vital.utils.image.measure import Measure, T

PixelCoord = Tuple[int, int]

logger = logging.getLogger(__name__)


class EchoMeasure(Measure):
    """Implementation of various echocardiography-specific measures on images."""

    #: 3x3 square structure, used by libraries like scipy and scikit-image to define the connectivity between labels.
    _square_3x3_connectivity = morphology.square(3)

    @staticmethod
    def _lv_base(
        segmentation: np.ndarray,
        lv_labels: SemanticStructureId = Label.LV.value,
        myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> Tuple[PixelCoord, PixelCoord]:
        """Identifies the coordinates of the left and right markers at the base of the left ventricle.

        The coordinates of the left ventricle's base is assumed to correspond to the bottom-left and bottom-right edges
        of the left ventricle/myocardium frontier.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.

        Returns:
            Coordinates of the left and right markers at the base of the left ventricle, or NaNs if the landmarks
            cannot be reliably estimated.
        """
        left_ventricle = np.isin(segmentation, lv_labels)
        myocardium = np.isin(segmentation, myo_labels)
        others = ~(left_ventricle + myocardium)
        dilated_myocardium = ndimage.binary_dilation(myocardium, structure=EchoMeasure._square_3x3_connectivity)
        dilated_others = ndimage.binary_dilation(others, structure=EchoMeasure._square_3x3_connectivity)
        y_coords, x_coords = np.where(left_ventricle * dilated_myocardium * dilated_others)

        if (num_markers := len(y_coords)) < 2:
            logger.warning(
                f"Identified {num_markers} marker(s) at the edges of the left ventricle/myocardium frontier. We need "
                f"to identify at least 2 such markers to determine the base of the left ventricle."
            )
            return np.nan, np.nan

        left_half_mask = x_coords < x_coords.mean()
        bottom_left_point_idx = y_coords[left_half_mask].argmax()
        bottom_right_point_idx = y_coords[~left_half_mask].argmax()
        return (
            (y_coords[left_half_mask][bottom_left_point_idx], x_coords[left_half_mask][bottom_left_point_idx]),
            (y_coords[~left_half_mask][bottom_right_point_idx], x_coords[~left_half_mask][bottom_right_point_idx]),
        )

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_base_width(
        segmentation: T,
        lv_labels: SemanticStructureId = Label.LV.value,
        myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> T:
        """Measures the distance between the left and right markers at the base of the left ventricle.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium. The location of the myocardium is
                necessary to identify the markers at the base of the left ventricle.

        Returns:
            ([N], 1), Distance between the left and right markers at the base of the left ventricle, or NaNs for the
            images where those 2 points cannot be reliably estimated.
        """
        lv_base_coords = EchoMeasure._lv_base(segmentation, lv_labels=lv_labels, myo_labels=myo_labels)
        return (
            np.linalg.norm(np.array(lv_base_coords[0]) - np.array(lv_base_coords[1]))
            if not np.isnan(lv_base_coords).any()
            else np.nan
        )

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_length(
        segmentation: T,
        lv_labels: SemanticStructureId = Label.LV.value,
        myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> T:
        """Measures the LV length as the distance between the LV's base midpoint and its furthest point at the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium. The location of the myocardium is
                necessary to identify the markers at the base of the left ventricle.

        Returns:
            ([N], 1), Length of the left ventricle, or NaNs for the images where the LV base's midpoint cannot be
            reliably estimated.
        """
        # Identify the base of the left ventricle
        lv_base_coords = EchoMeasure._lv_base(segmentation, lv_labels=lv_labels, myo_labels=myo_labels)
        if np.isnan(lv_base_coords).any():
            # Early return if we couldn't reliably estimate the landmarks at the base of the left ventricle
            return np.nan

        # Identify the midpoint at the base of the left ventricle
        lv_base_mid = np.array(lv_base_coords).mean(axis=0)

        # Compute the distance from all pixels in the image to `lv_base_midpoint`
        mask = np.ones_like(segmentation, dtype=bool)
        mask[tuple(lv_base_mid.round().astype(int))] = False
        dist_to_lv_base_mid = ndimage.distance_transform_edt(mask)

        # Find the point within the left ventricle mask with maximum distance
        left_ventricle = np.isin(segmentation, lv_labels)
        lv_apex_coords = np.unravel_index(np.argmax(dist_to_lv_base_mid * left_ventricle), segmentation.shape)

        return np.linalg.norm(np.array(lv_apex_coords) - lv_base_mid)
