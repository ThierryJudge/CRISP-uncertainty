import numpy as np
from skimage.measure import regionprops

from vital.data.camus.config import Label
from vital.utils.format.numpy import to_categorical
from vital.utils.image.measure import Measure
from vital.utils.image.register.affine import AffineRegisteringTransformer, Crop, Rotation, Shift


class CamusRegisteringTransformer(AffineRegisteringTransformer):
    """Class that allows to register CAMUS dataset's image/segmentation pairs.

    The goal of this registration is to shift the centroid of the epicardium in the middle of the image and rotate the
    image so the major axis of the left ventricle is vertically aligned.

    It inherits the image transformation utilities from `AffineRegisteringTransformer`, and only has to implement the
    algorithsm for finding the affine registering parameters. These parameters are obtained based on the content of the
    segmentation maps and not the input MRI image.
    """

    registering_steps = ["shift", "rotation", "crop"]

    def _compute_shift_parameters(self, segmentation: np.ndarray) -> Shift:
        """Computes the pixel shift along each axis to center the segmentation around the epicardium structure.

        Args:
            segmentation: Segmentation for which to compute shift parameters.

        Returns:
            Pixel shift along each axis to center the segmentation around the epicardium structure.
        """
        # Find the bottom limit of the atrium
        # (using a bounding box encompassing all segmentation classes)
        segmentation_mask = 1 - segmentation[..., Label.BG.value]
        segmentation_props = regionprops(segmentation_mask)[0]
        distance_from_left_atrium_to_border = segmentation.shape[0] - segmentation_props.bbox[2]

        # Find the center of mass of the epicardium (union of the left ventricle and myocardium)
        segmentation_center = segmentation.shape[0] // 2, segmentation.shape[1] // 2
        epicardium_center = Measure.structure_center(to_categorical(segmentation), [Label.LV.value, Label.MYO.value])

        # Center the image as closely as possible around the epicardium without cutting off the left atrium
        rows_shift = max(epicardium_center[0] - segmentation_center[0], -distance_from_left_atrium_to_border)
        columns_shift = epicardium_center[1] - segmentation_center[1]
        return rows_shift, columns_shift

    def _compute_rotation_parameters(self, segmentation: np.ndarray) -> Rotation:
        """Computes the rotation to align the major axis of the left ventricle ellipse with the vertical axis.

        Args:
            segmentation: Segmentation for which to compute rotation parameters.

        Returns:
            Angle of the rotation to align the major axis of the left ventricle with the vertical axis.
        """
        return Measure.structure_orientation(
            to_categorical(segmentation), Label.LV.value, reference_orientation=90
        ).item()

    def _compute_crop_parameters(self, segmentation: np.ndarray, margin: float = 0.05) -> Crop:
        """Computes the coordinates of an isotropic bounding box around all segmented classes.

        Args:
            segmentation: Segmentation for which to compute crop parameters.
            margin: Ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight margin at
                the edges of the bbox.

        Returns:
            Original shape and coordinates of the bbox, in the following order:
            height, width, row_min, col_min, row_max, col_max.
        """
        # Get the best fitting bbox around the segmented structures
        bbox_coord = Measure.bbox(segmentation, list(range(1, self.num_classes)), bbox_margin=margin)
        bbox_shape = (bbox_coord[2] - bbox_coord[0], bbox_coord[3] - bbox_coord[1])

        # Find the parameters to get a square bbox from the current, best fit, rectangular bbox
        shape_diff = abs(bbox_shape[1] - bbox_shape[0])
        smallest_dim = np.argmin(bbox_shape)

        # Elongate the smallest side of the rectangular bbox so that it is now as square as possible
        # While ensuring that they we don't land outside the array
        bbox_coord[smallest_dim] = bbox_coord[smallest_dim] - shape_diff // 2
        bbox_coord[smallest_dim + 2] = bbox_coord[smallest_dim + 2] + shape_diff // 2

        return segmentation.shape[:2] + tuple(bbox_coord)

    # Uncomment following function to enable "zoom to fit" registration step
    #
    # def _compute_zoom_to_fit_parameters(self, segmentation: np.ndarray, margin: float = 0.1) -> Zoom:
    #     """Computes the zoom along each axis to fit the bounding box surrounding the segmented classes.
    #
    #     Args:
    #         segmentation: Segmentation for which to compute zoom to fit parameters.
    #         margin: Ratio of image shape to ignore when computing zoom so as to leave empty border around the image
    #             when fitting.
    #
    #     Returns:
    #         Zoom along each axis to fit the bounding box surrounding the segmented classes.
    #     """
    #     # Find dimensions of the bounding box encompassing all segmentation classes
    #     segmentation_mask = (
    #         segmentation[..., Label.LV.value]
    #         | segmentation[..., Label.MYO.value]
    #         | segmentation[..., Label.ATRIUM.value]
    #     )
    #     segmentation_props = regionprops(segmentation_mask)[0]
    #     segmentation_bbox = segmentation_props.bbox
    #
    #     # Use height zoom factor along each axis because it is the limiting factor and we want to keep proportions
    #     segmentation_bbox_height = segmentation_bbox[2] - segmentation_bbox[0]
    #     zoom = segmentation_bbox_height / (segmentation.shape[0] * (1 - margin))
    #     return zoom, zoom
