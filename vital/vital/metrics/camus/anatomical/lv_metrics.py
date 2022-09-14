from vital.data.camus.config import Label
from vital.metrics.evaluate.anatomical_structure import Anatomical2DStructureMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class LeftVentricleMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the left ventricle."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.LV.value)
