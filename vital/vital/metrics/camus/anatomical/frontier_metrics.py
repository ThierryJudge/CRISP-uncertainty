from vital.data.camus.config import Label
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class FrontierMetrics:
    """Class to compute metrics on the frontiers between segmentations of multiple anatomical structures."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics

    def count_holes_between_lv_and_myo(self) -> int:
        """Counts the pixels in the gap between the left ventricle (LV) and myocardium (MYO).

        Returns:
            Count of pixels in the gap between the two areas segmented as left ventricle and myocardium.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.LV.value, Label.MYO.value)

    def count_holes_between_lv_and_atrium(self) -> int:
        """Counts the pixels in the gap between the left ventricle (LV) and left atrium.

        Returns:
            Count of pixels in the gap between the two areas segmented as left ventricle and left atrium.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.LV.value, Label.MYO.value)

    def measure_frontier_ratio_between_lv_and_bg(self) -> float:
        """Measures the ratio between the length of the frontier between the LV and BG and the width of the LV.

        Returns:
            Ratio between the length of the frontier between the LV and BG and the width of the LV.
        """
        return self.segmentation_metrics.measure_frontier_ratio_between_regions(Label.LV.value, Label.BG.value)

    def measure_frontier_ratio_between_myo_and_atrium(self) -> float:
        """Measures the ratio between the length of the frontier between the MYO and atrium and the width of the MYO.

        Returns:
            Ratio between the length of the frontier between the MYO and atrium and the width of the MYO.
        """
        return self.segmentation_metrics.measure_frontier_ratio_between_regions(Label.MYO.value, Label.ATRIUM.value)
