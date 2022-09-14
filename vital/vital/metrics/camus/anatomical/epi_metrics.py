from vital.data.camus.config import Label
from vital.metrics.evaluate.anatomical_structure import Anatomical2DStructureMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class EpicardiumMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the left ventricle epicardium (LV + MYO)."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        # For myocardium we want to calculate anatomical metrics for the entire epicardium
        # Therefore we concatenate label 1 (lumen) and 2 (myocardium)
        super().__init__(segmentation_metrics, (Label.LV.value, Label.MYO.value))
