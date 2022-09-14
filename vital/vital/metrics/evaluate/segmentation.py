from numbers import Real
from typing import Literal, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, measurements
from skimage import measure, morphology
from skimage.morphology import convex_hull_image

from vital.data.config import SemanticStructureId


class Segmentation2DMetrics:
    """Wraps a segmentation map and implements algorithms that operate on it to compute metrics."""

    def __init__(
        self,
        segmentation: np.ndarray,
        struct_labels: Sequence[SemanticStructureId],
        voxelspacing: Tuple[Real, Real] = (1.0, 1.0),
    ):
        """Initializes class instance.

        Args:
            segmentation: (H, W), 2D array where the value of each entry in the array is the label of the segmentation
                class for the pixel.
            struct_labels: Label(s) of the class(es) present in the segmentation for which to compute metrics.
            voxelspacing: Size of the voxels along each (height, width) dimension (in mm).
        """
        self.segmentation = segmentation
        self.voxelspacing = voxelspacing
        self.binary_structs = {
            struct_label: np.isin(segmentation, struct_label).astype(dtype=np.uint8) for struct_label in struct_labels
        }
        self.binary_structs_inverse = {
            struct_label: 1 - binary_struct for struct_label, binary_struct in self.binary_structs.items()
        }

        # Approximate size of small blobs determined through empirical experiments. Those small blobs correspond to
        # small artifacts left behind by some morphological operations on the images, and which can be ignored since
        # they would otherwise skew the metrics' values.
        self.small_objects_size = segmentation.shape[0] * segmentation.shape[1] / 2 ** 14

    def count_holes(self, struct_label: SemanticStructureId) -> int:
        """Counts the pixels that form holes in a supposedly contiguous segmented area.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.

        Returns:
            Number of pixels that form holes in the segmented area.
        """
        # Mark the class of interest as 0 and everything else as 1
        # Merge the regions of 1 that are open by a side using padding (these are not holes)
        binary_struct = self.binary_structs_inverse[struct_label]
        binary_struct = np.pad(binary_struct, ((1, 1), (1, 1)), "constant", constant_values=1)

        # Extract properties of continuous regions of 1
        props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        hole_pixel_count = 0
        for prop in props:
            # Skip the region open by the side (the one that includes padding)
            if prop.bbox[0] != 0:
                hole_pixel_count += prop.area

        return hole_pixel_count

    def count_disconnectivity(self, struct_label: SemanticStructureId) -> int:
        """Counts the pixels that are disconnected from a supposedly contiguous segmented area.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.

        Returns:
            Total number of disconnected pixels in the segmented area.
        """
        binary_struct = self.binary_structs[struct_label]

        # Extract properties for every disconnected region of the supposedly continuous segmented area
        labels_props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        if len(labels_props) > 1:  # If there is disconnectivity in the segmentated area
            # Sort by decreasing pixel count of the region
            labels_props_by_desc_size = sorted(labels_props, key=lambda k: k.area, reverse=True)

            total_minus_biggest = sum(label_props.area for label_props in labels_props_by_desc_size[1:])
            return total_minus_biggest

        else:  # If there was only a single contiguous region making up the segmented area
            return 0

    def count_holes_between_regions(
        self, struct1_label: SemanticStructureId, struct2_label: SemanticStructureId
    ) -> int:
        """Counts the pixels in the gap between two supposedly connected segmented areas.

        Warnings:
            - As this method iterates over holes (directly in Python) in supposedly filled-in segmented areas, its
              runtime will increase by multiple orders of magnitude if tested on degenerated results (e.g. similar to
              noise), where segmented areas are not filled-in and can be seen as full of a multitude of tiny holes.

        Args:
            struct1_label: Label(s) of the class(es) making up one of the segmented areas.
            struct2_label: Label(s) of the class(es) making up the other segmented area.

        Returns:
            Number of pixels in the gap between the two segmented areas.
        """
        # Obtain binary segmentations for each segmented areas
        binary_struct1 = self.binary_structs[struct1_label]
        binary_struct2 = self.binary_structs[struct2_label]
        not_binary_struct1 = self.binary_structs_inverse[struct1_label]
        not_binary_struct2 = self.binary_structs_inverse[struct2_label]

        # Find holes inside the first segmented area
        holes_struct1, last_hole_label_class1 = measure.label(not_binary_struct1, connectivity=1, return_num=True)
        outside_class1 = holes_struct1[0, 0]
        holes_struct1[holes_struct1 == outside_class1] = 0

        # Find holes inside the second segmented area
        holes_class2, last_hole_label_class2 = measure.label(not_binary_struct2, connectivity=1, return_num=True)
        outside_class2 = holes_class2[0, 0]
        holes_class2[holes_class2 == outside_class2] = 0

        # Find the holes in the first area that don't contain the second area (and the opposite)
        hole_labels_class1_without_class2 = [
            h for h in range(1, last_hole_label_class1 + 1) if h not in np.unique(holes_struct1 * binary_struct2)
        ]
        hole_labels_class2_without_class1 = [
            h for h in range(1, last_hole_label_class2 + 1) if h not in np.unique(holes_class2 * binary_struct1)
        ]
        holes_class1_without_class2 = np.isin(
            holes_struct1, hole_labels_class1_without_class2, assume_unique=True
        ).astype(dtype=np.uint8)
        holes_class2_without_class1 = np.isin(
            holes_class2, hole_labels_class2_without_class1, assume_unique=True
        ).astype(dtype=np.uint8)

        not_both = 1 - (
            (binary_struct1 + holes_class1_without_class2 + binary_struct2 + holes_class2_without_class1) > 0
        )
        holes_between = measure.label(not_both, connectivity=1)
        outside = holes_between[0, 0]
        holes_between[holes_between == outside] = 0
        pixels_in_holes = np.sum(holes_between > 0)

        return pixels_in_holes if pixels_in_holes > self.small_objects_size else 0

    def count_frontier_between_regions(
        self, struct1_label: SemanticStructureId, struct2_label: SemanticStructureId
    ) -> int:
        """Counts the pixels touching between two supposedly disconnected segmented areas.

        Args:
            struct1_label: Label(s) of the class(es) making up one of the segmented areas.
            struct2_label: Label(s) of the class(es) making up the other segmented area.

        Returns:
            Number of pixels on the frontier between the two segmented areas.
        """
        # Obtain binary segmentations for each segmented area
        binary_struct1 = self.binary_structs[struct1_label]
        binary_struct2 = self.binary_structs[struct2_label]
        struct2_dilated = morphology.dilation(binary_struct2, np.ones((3, 3)))
        frontier = binary_struct1 * struct2_dilated
        pixels_on_frontier = frontier.sum()
        return pixels_on_frontier if pixels_on_frontier > 1 else 0

    def measure_concavity(self, struct_label: SemanticStructureId, no_structure_flag: float = float("nan")) -> float:
        """Measures the depth of a concavity in a supposedly convex segmented area.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.
            no_structure_flag: Value to return when no structure was found in the segmentation.

        Returns:
             Depth (in mm) of a concavity in the segmented area.
        """
        # Use binary structure with holes filled in
        # Necessary for structures that are not supposed to filled (e.g. myocardium)
        binary_struct = binary_fill_holes(self.binary_structs[struct_label]).astype(dtype=np.uint8)

        if np.amax(binary_struct):  # If the structure is present in the image

            # Get convex hull of the segmented area
            convex_hull = morphology.convex_hull_image(binary_struct)

            # Compute a mask for the difference between the convex hull and the source image
            # The mask is eroded to account for the behavior of ```distance_transform_edt``` where a pixel
            # on the frontier with the background is at distance 1 from the background
            diff_convex_img = convex_hull - binary_struct
            diff_convex_img = morphology.binary_erosion(diff_convex_img, np.ones((3, 3)))

            # Compute distance from convex hull pixels to nearest background pixel
            convex_dist_to_background = distance_transform_edt(convex_hull, sampling=self.voxelspacing)

            # Apply the convex hull difference mask to the distances
            convex_dist_to_background *= diff_convex_img

            # Get the Hausdorff distance by finding the maximum distance in the remaining distances
            hausdroff_distance = np.max(convex_dist_to_background)
            return hausdroff_distance

        else:  # If the structure is not in the image
            return no_structure_flag

    def measure_circularity(self, struct_label: SemanticStructureId, no_structure_flag: float = float("nan")) -> float:
        """Measures the isoperimetric ratio of a segmented area, assuming the area is contiguous.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.
            no_structure_flag: Value to return when no structure was found in the segmentation.

        Returns:
            Isoperimetric ratio of the segmented area.
        """
        # Use binary structure with holes filled in
        # Necessary for structures that are not supposed to filled (e.g. myocardium)
        binary_struct = binary_fill_holes(self.binary_structs[struct_label]).astype(dtype=np.uint8)

        if np.amax(binary_struct):  # If the structure is present in the image
            # Compute the perimeter of the given class
            perimeter = measure.perimeter(binary_struct, neighbourhood=4)
            if perimeter < 1e-6:
                return no_structure_flag

            # Compute the area of the given class
            area = measure.moments(binary_struct)[0, 0]
            if area < 1e-6:
                return no_structure_flag

            # Compute the isoperimetric ratio ( https://en.wikipedia.org/wiki/Isoperimetric_inequality )
            ratio = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            return ratio

        else:  # If the structure is not in the image
            return no_structure_flag

    def measure_frontier_ratio_between_regions(
        self, struct1_label: SemanticStructureId, struct2_label: SemanticStructureId
    ) -> float:
        """Measures the ratio between the length of the frontier between two structures and the size of the first one.

        Notes:
            - The frontier between the two structures is considered as the longest continuous segment where pixels
              belonging to both structures are adjacent.
            - Because the frontier length is rationed over the width of the first structure, the arguments
              ``struct_label1`` and ``struct_label2`` are not interchangeable.

        Args:
            struct1_label: Label(s) of the class(es) making up one of the segmented areas.
            struct2_label: Label(s) of the class(es) making up the other segmented area.

        Returns:
            Ratio between the largest continuous segment of the frontier and the width of the first segmented area.
        """
        # Measure the number of pixels on the frontier
        pixels_on_frontier = self.count_frontier_between_regions(struct1_label, struct2_label)

        # Measure the length (in pixel) of the width of the first segmented area
        binary_struct1 = self.binary_structs[struct1_label]
        if np.amax(binary_struct1):  # If the structure is present in the image
            binary_struct1_bbox = measure.regionprops(binary_struct1)[0].bbox
            binary_struct1_width = binary_struct1_bbox[3] - binary_struct1_bbox[1]  # 1 = bbox min col, 3 = bbox max col

            # Find the ratio between the largest continuous segment of the frontier and the width of the first segmented
            # area
            frontier_proportion = pixels_on_frontier / binary_struct1_width
            return frontier_proportion

        else:  # If the structure is not in the image
            return 0

    def measure_width_ratio_between_regions(
        self,
        struct1_label: SemanticStructureId,
        struct2_label: SemanticStructureId,
        no_structure_flag: float = float("nan"),
    ) -> float:
        """Measures the ratio between the width of two structures at the center of mass of the regions.

        Notes:
            - The width is not necessarily contiguous, and is measured by summing along the segment belonging to
              structure along the chosen axis.

        Args:
            struct1_label: Label(s) of the class(es) making up one of the segmented areas.
            struct2_label: Label(s) of the class(es) making up the other segmented area.
            no_structure_flag: Value to return if either of the structure is not found in the segmentation.

        Returns:
            Ratio between the width (not necessarily contiguous) of two structures at the center of mass of the regions.
        """
        # Find center of mass and measure width ratio there instead of the center of the image
        # This is to avoid cases where the structures to segment are small and in one corner of the image

        # Compute the center of mass
        joint_binary_structs = np.logical_or(self.binary_structs[struct1_label], self.binary_structs[struct2_label])
        center_of_mass_row = int(measurements.center_of_mass(joint_binary_structs)[0])

        # Measure the width ratio at the center of mass
        struct1_width = np.sum(self.binary_structs[struct1_label][center_of_mass_row])
        struct2_width = np.sum(self.binary_structs[struct2_label][center_of_mass_row])
        return struct1_width / struct2_width if (struct1_width and struct2_width) else no_structure_flag

    def measure_erosion_ratio_before_split(
        self, struct_label: SemanticStructureId, no_structure_flag: float = float("nan")
    ) -> float:
        """Measures the relative difference between the thickness of a structure at its narrowest and widest points.

        The thickness comparison is done by measuring the ratio between the depth of erosion necessary to split a
        continuous anatomical structure and the maximum thickness (in pixels) of the structure.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.
            no_structure_flag: Value to return when no structure was found in the segmentation.

        Returns:
            Ratio between the depth of erosion necessary to divide a continuous structure in at least two fragments and
            the maximum thickness (in pixels) of the structure.
        """
        binary_struct = self.binary_structs[struct_label]
        if np.amax(binary_struct):  # If the structure is present in the image

            # Compute maximum thickness of the structure (pixel with maximum distance to nearest background pixel)
            struct_dist_to_background = distance_transform_edt(binary_struct)
            max_struct_dist_to_background = np.max(struct_dist_to_background)

            # Erode the continuous structure until it splits in at least two fragments
            eroding_binary_struct = binary_struct
            nb_erosions = 0
            while measure.label(eroding_binary_struct, connectivity=2, return_num=True)[1] == 1:
                eroding_binary_struct = morphology.binary_erosion(eroding_binary_struct, np.ones((3, 3)))
                morphology.remove_small_objects(eroding_binary_struct, min_size=self.small_objects_size, in_place=True)
                nb_erosions += 1

            return nb_erosions / max_struct_dist_to_background

        else:  # If the structure is not in the image
            return no_structure_flag

    def measure_convexity(self, struct_label: SemanticStructureId, no_structure_flag: float = float("nan")) -> float:
        """Measures the shape convexity of a segmented area.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.
            no_structure_flag: Value to return when no structure was found in the segmentation.

        Returns:
            Value of the shape convexity metric for the segmented area.
        """
        binary_struct = self.binary_structs[struct_label]
        if np.amax(binary_struct):  # If the structure is present in the image
            struct_convex_hull = convex_hull_image(binary_struct)
            convexity = np.sum(binary_struct) / np.sum(struct_convex_hull)
            return convexity

        else:  # If the structure is not in the image
            return no_structure_flag


def check_metric_validity(
    metric_value: Real, thresholds: Mapping[Literal["min", "max"], Real] = None, optional_structure: bool = False
) -> bool:
    """Checks whether the value of the metric is within the range of values meaning the segmentation is correct.

    Args:
        metric_value: Value of the metric for a segmentation.
        thresholds: Minimum ('min') and maximum ('max') tolerated value for the metric. Any value below the minimum or
            above the maximum will lead to the segmentation being considered erroneous. If no thresholds are specified,
            a value of '0' is considered to mean no error, and any other value means an error.
        optional_structure: If ``True``, the segmentation can be considered valid even if the metric' value indicates
            the structure on which it was measured is absent. Otherwise, a missing structure is considered invalid.

    Returns:
        ``True`` if the metric is within the threshold values (i.e. the segmentation is valid). ``False`` otherwise.
    """
    if np.isnan(metric_value):  # If the structure on which to compute the metric was not present in the segmentation
        validity = optional_structure
    elif thresholds:  # If the metric has a range of value within which segmentations are considered valid
        validity = thresholds.get("min", 0) <= metric_value <= thresholds.get("max", np.inf)
    else:  # If no range of valid values are provided, a value of '0' is considered to mean no error, and any other
        # value means an error
        validity = not bool(metric_value)
    return validity
