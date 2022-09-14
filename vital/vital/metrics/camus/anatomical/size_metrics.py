from vital.data.camus.config import Label
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class SizeMetrics:
    """Class to compute metrics comparing sizes between segmentations of multiple anatomical structures."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics
        self.no_structure_flag = float("nan")

    def measure_width_ratio_between_lv_and_myo(self) -> float:
        """Measures the relative width of the left ventricle and the myocardium at their joined center of mass.

        The width comparison is measured using the ratio between the width the width of the left ventricle and the total
        width of both myocardium walls along an horizontal line anchored at their joined center of mass.

        Returns:
            Ratio between the width the width of the left ventricle and the total width of both myocardium walls along
            an horizontal line anchored at their joined center of mass.
        """
        return self.segmentation_metrics.measure_width_ratio_between_regions(
            Label.LV.value, Label.MYO.value, no_structure_flag=self.no_structure_flag
        )
