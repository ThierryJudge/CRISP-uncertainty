import math
from numbers import Real
from typing import Tuple, TypeVar

import numpy as np
from scipy import ndimage
from skimage import measure
from torch import Tensor

from vital.data.config import SemanticStructureId
from vital.utils.decorators import auto_cast_data, batch_function

T = TypeVar("T", np.ndarray, Tensor)


class Measure:
    """Generic implementations of various measures on images represented as numpy arrays or torch tensors."""

    @staticmethod
    @auto_cast_data
    def structure_area(segmentation: T, labels: SemanticStructureId) -> T:
        """Computes the number of pixels, in a segmentation map, associated to a structure.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the number of pixels of the structure.
            labels: Labels of the classes that are part of the structure for which to count the number of pixels.

        Returns:
            ([N], 1), Number of pixels associated to the structure, in each segmentation of the batch.
        """
        return np.isin(segmentation, labels).sum((-2, -1))[..., None]

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def structure_center(segmentation: T, labels: SemanticStructureId, axis: int = None) -> T:
        """Computes the center of mass of a structure in a segmentation map.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the center of mass of the structure.
            labels: Labels of the classes that are part of the structure for which to measure the center of mass.
            axis: Index of a dimension of interest for which to get the center of mass. If provided, the value of the
                center of mass will only be returned for this axis. If `None`, the center of mass along all axes will be
                returned.

        Returns:
            ([N], {1|2}), Center of mass of the structure, for a specified axis or across all axes, in each segmentation
            of the batch.
        """
        center = ndimage.measurements.center_of_mass(np.isin(segmentation, labels))
        if any(np.isnan(center)):  # Default to the center of the image if the center of mass can't be found
            center = np.array(segmentation.shape) // 2
        if axis is not None:
            center = center[axis]
        return center

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def structure_orientation(segmentation: T, labels: SemanticStructureId, reference_orientation: int = 0) -> T:
        """Computes the angle w.r.t. a reference orientation of a structure in a segmentation map.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the orientation of the structure.
            labels: Labels of the classes that are part of the structure for which to measure orientation.
            reference_orientation: Reference orientation, that would correspond to a returned orientation of `0` if the
                structure where aligned on it perfectly. By default, this orientation corresponds to the positive x
                axis.

        Returns:
            ([N], 1), Orientation of the structure w.r.t. the reference orientation, in each segmentation of the batch.
        """
        structure_mask = np.isin(segmentation, labels)
        if np.any(structure_mask):  # If the structure is present in the segmentation
            # Get the right eigenvectors of the structure's mass
            structure_inertia_tensors = measure.inertia_tensor(structure_mask)
            _, evecs = np.linalg.eigh(structure_inertia_tensors)

            # Find the 1st eigenvector, that corresponds to the orientation of the structure's longest axis
            evec1 = evecs[-1]

            # Compute the rotation necessary to align it with the x-axis (horizontal)
            orientation = math.degrees(np.arctan2(evec1[1], evec1[0]))
            orientation -= reference_orientation  # Get angle with reference orientation from angle with x-axis
        else:  # If the structure is not present in the segmentation, consider it aligned to the reference by default
            orientation = 0
        return orientation

    @staticmethod
    @auto_cast_data
    def bbox(segmentation: T, labels: SemanticStructureId, bbox_margin: Real = 0.05, normalize: bool = False) -> T:
        """Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the coordinates of the bbox.
            labels: Labels of the classes that are part of the ROI.
            bbox_margin: Ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight
                margin at the edges of the bbox.
            normalize: If ``True``, normalizes the bbox coordinates from between 0 and H or W to between 0 and 1.

        Returns:
            ([N], 4), Coordinates of the bbox, in (y1, x1, y2, x2) format.
        """
        # Only keep ROI from the groundtruth
        roi_mask = np.isin(segmentation, labels)

        # Find the coordinates of the bounding box around the ROI
        rows = roi_mask.any(1)
        cols = roi_mask.any(0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Compute the size of the margin between the ROI and its bounding box
        dx = int(bbox_margin * (x2 - x1))
        dy = int(bbox_margin * (y2 - y1))

        # Apply margin to bbox coordinates
        y1, y2 = y1 - dy, y2 + dy + 1
        x1, x2 = x1 - dx, x2 + dx + 1

        # Check limits
        y1, y2 = max(0, y1), min(y2, roi_mask.shape[0])
        x1, x2 = max(0, x1), min(x2, roi_mask.shape[1])

        roi_bbox = np.array([y1, x1, y2, x2])

        if normalize:
            roi_bbox = roi_bbox.astype(float)
            roi_bbox[[0, 2]] = roi_bbox[[0, 2]] / segmentation.shape[0]  # Normalize height
            roi_bbox[[1, 3]] = roi_bbox[[1, 3]] / segmentation.shape[1]  # Normalize width

        return roi_bbox

    @staticmethod
    @auto_cast_data
    def denormalize_bbox(roi_bbox: T, output_size: Tuple[int, int], check_bounds: bool = False) -> T:
        """Gives the pixel-indices of a bounding box (bbox) w.r.t an output size based on the bbox's normalized coord.

        Args:
            roi_bbox: ([N], 4), Normalized coordinates of the bbox, in (y1, x1, y2, x2) format.
            output_size: (X, Y), Size for which to compute pixel-indices based on the normalized coordinates.
            check_bounds: If ``True``, perform various checks on the denormalized coordinates:
                - ensure they fit between 0 and X or Y
                - ensure that the min bounds are smaller than the max bounds
                - ensure that the bbox is at least one pixel wide in each dimension

        Returns:
            ([N], 4), Coordinates of the bbox, in (y1, x1, y2, x2) format.
        """
        # Copy input data to ensure we don't write over user data
        roi_bbox = np.copy(roi_bbox)

        if check_bounds:
            # Clamp predicted RoI bbox to ensure it won't end up out of range of the image
            roi_bbox = np.clip(roi_bbox, 0, 1)

        # Change ROI bbox from normalized between 0 and 1 to absolute pixel coordinates
        roi_bbox[:, (0, 2)] = (roi_bbox[:, (0, 2)] * output_size[0]).round()  # Y
        roi_bbox[:, (1, 3)] = (roi_bbox[:, (1, 3)] * output_size[1]).round()  # X

        if check_bounds:
            # Clamp predicted min bounds are at least two pixels smaller than image bounds
            # to allow for inclusive upper bounds
            roi_bbox[:, 0] = np.minimum(roi_bbox[:, 0], output_size[0] - 1)  # Y
            roi_bbox[:, 1] = np.minimum(roi_bbox[:, 1], output_size[1] - 1)  # X

            # Clamp predicted max bounds are at least one pixel bigger than min bounds
            roi_bbox[:, 2] = np.maximum(roi_bbox[:, 2], roi_bbox[:, 0] + 1)  # Y
            roi_bbox[:, 3] = np.maximum(roi_bbox[:, 3], roi_bbox[:, 1] + 1)  # X

        return roi_bbox
