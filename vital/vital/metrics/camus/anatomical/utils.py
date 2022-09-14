from numbers import Real
from typing import Dict, Sequence, Tuple

import numpy as np

from vital.data.camus.config import Label
from vital.metrics.camus.anatomical.atrium_metrics import LeftAtriumMetrics
from vital.metrics.camus.anatomical.config import thresholds
from vital.metrics.camus.anatomical.epi_metrics import EpicardiumMetrics
from vital.metrics.camus.anatomical.frontier_metrics import FrontierMetrics
from vital.metrics.camus.anatomical.lv_metrics import LeftVentricleMetrics
from vital.metrics.camus.anatomical.myo_metrics import MyocardiumMetrics
from vital.metrics.camus.anatomical.size_metrics import SizeMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics, check_metric_validity


def check_segmentation_validity(
    segmentation: np.ndarray, voxelspacing: Tuple[Real, Real], labels: Sequence[Label] = None, **kwargs
) -> bool:
    """Checks that the segmentation passes a set of predefined anatomical criteria.

    Args:
        segmentation: Segmentation for which to compute anatomical metrics.
        voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).
        labels: Labels of the classes included in the segmentation.
        **kwargs: Catch additional parameters for API compatibility with similar functions for anatomical validity
            across multiple datasets.

    Returns:
        ``True`` if the segmentation is anatomically plausible (i.e. it passes all the anatomical criteria);
        ``False`` otherwise.
    """
    return not compute_anatomical_metrics_by_segmentation(segmentation, voxelspacing, labels=labels)[
        "anatomical_errors"
    ]


def compute_anatomical_metrics_by_segmentation(
    segmentation: np.ndarray, voxelspacing: Tuple[Real, Real], labels: Sequence[Label] = None
) -> Dict[str, Real]:
    """Computes the anatomical metrics for a single segmentation.

    Args:
        segmentation: Segmentation as a 2d array.
        voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).
        labels: Labels of the classes included in the segmentation.

    Returns:
        Mapping between the anatomical metrics' names and their value for the segmentation.
    """
    if labels is None:
        labels = list(Label)
    # For myocardium we want to calculate anatomical metrics for the entire epicardium
    # Therefore we concatenate label 1 (lumen) and 2 (myocardium)
    segmentation_metrics = Segmentation2DMetrics(
        segmentation,
        [Label.BG.value, Label.LV.value, Label.MYO.value, (Label.LV.value, Label.MYO.value), Label.ATRIUM.value],
        voxelspacing=voxelspacing,
    )
    lv_metrics = LeftVentricleMetrics(segmentation_metrics)
    myo_metrics = MyocardiumMetrics(segmentation_metrics)
    atrium_metrics = LeftAtriumMetrics(segmentation_metrics)
    epi_metrics = EpicardiumMetrics(segmentation_metrics)
    frontier_metrics = FrontierMetrics(segmentation_metrics)
    size_metrics = SizeMetrics(segmentation_metrics)
    metrics = {}
    if Label.LV in labels:
        metrics.update(
            {
                "holes_in_lv": lv_metrics.count_holes(),
                "disconnectivity_in_lv": lv_metrics.count_disconnectivity(),
            }
        )
    if Label.MYO in labels:
        metrics.update(
            {
                "disconnectivity_in_myo": myo_metrics.count_disconnectivity(),
                "erosion_ratio_before_myo_split": myo_metrics.measure_erosion_ratio_before_split(),
            }
        )
    if Label.ATRIUM in labels:
        metrics.update(
            {
                "holes_in_atrium": atrium_metrics.count_holes(),
                "disconnectivity_in_atrium": atrium_metrics.count_disconnectivity(),
            }
        )
    if (Label.LV in labels) and (Label.MYO in labels):
        metrics.update(
            {
                "holes_in_epi": epi_metrics.count_holes(),
                "holes_between_lv_and_myo": frontier_metrics.count_holes_between_lv_and_myo(),
                "width_ratio_between_lv_and_myo": size_metrics.measure_width_ratio_between_lv_and_myo(),
            }
        )
    if (Label.ATRIUM in labels) and (Label.LV in labels):
        metrics.update(
            {
                "holes_between_lv_and_atrium": frontier_metrics.count_holes_between_lv_and_atrium(),
                "frontier_ratio_between_lv_and_bg": frontier_metrics.measure_frontier_ratio_between_lv_and_bg(),
            }
        )
    if (Label.ATRIUM in labels) and (Label.MYO in labels):
        metrics.update(
            {"frontier_ratio_between_myo_and_atrium": frontier_metrics.measure_frontier_ratio_between_myo_and_atrium()}
        )
    metrics["anatomical_errors"] = any(
        not check_metric_validity(value, thresholds.get(name), optional_structure=False)
        for name, value in metrics.items()
    )
    return metrics
