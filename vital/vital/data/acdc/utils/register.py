from math import degrees

import numpy as np

from vital.data.acdc.config import Label
from vital.utils.image.register.affine import AffineRegisteringTransformer, Rotation, Shift


class AcdcRegisteringTransformer(AffineRegisteringTransformer):
    """Class that allows to register ACDC dataset's image/segmentation pairs.

    Implements algorithms for finding the affine registering parameters. These parameters are obtained based on the
    content of the segmentation maps and not the input MRI image. The goal of this registration is to shift the
    centroid of the left ventricle in the middle of the image and rotate the image so the centroid of the right
    ventricle is on the left of the centroid of the left ventricle.
    """

    registering_steps = ["shift", "rotation"]

    def __init__(self):
        super().__init__(len(Label))

    def _compute_shift_parameters(self, segmentation: np.ndarray) -> Shift:
        """Computes the pixel shift to apply along each axis to center the segmentation around the LV/MYO mass.

        Args:
            segmentation: segmentation for which to compute shift parameters.

        Returns:
            Pixel shift to apply along each axis to center the segmentation around the LV/MYO mass.
        """
        segmentation_center = np.array(segmentation.shape[:2]) // 2
        lv_myo_center = np.array(self._find_structure_center(segmentation, [Label.LV.value, Label.MYO.value]))
        return (lv_myo_center - segmentation_center).astype(int)

    def _compute_rotation_parameters(self, segmentation: np.ndarray) -> Rotation:
        """Computes the angle of the rotation.

        Angle is computed to apply to align the segmentation so that the RV center of mass is
        always left of the LV/MYO center of mass on a straight horizontal line.

        Args:
            segmentation: for which to compute rotation parameters.

        Returns:
            Angle of the rotation to apply to align the segmentation so that the RV center of mass is always
            left of the LV/MYO center of mass on a straight horizontal line.
        """
        lv_myo_center = np.array(self._find_structure_center(segmentation, [Label.LV.value, Label.MYO.value]))
        rv_center = np.array(
            self._find_structure_center(segmentation, Label.RV.value, default_center=tuple(lv_myo_center))
        )
        centers_diff = rv_center - lv_myo_center
        rotation_angle = degrees(np.arctan2(centers_diff[1], centers_diff[0]))
        rotation_angle = -(180 - ((rotation_angle + 360) % 360))
        return rotation_angle
