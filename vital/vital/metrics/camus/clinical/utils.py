from numbers import Real
from typing import Dict, Tuple

import numpy as np

from vital.data.camus.config import Label
from vital.metrics.evaluate.clinical.heart_us import compute_left_ventricle_volumes


def compute_clinical_metrics_by_patient(
    results: Tuple[np.ndarray, np.ndarray],
    references: Tuple[np.ndarray, np.ndarray],
    a2c_voxelspacing: Tuple[Real, Real],
    a4c_voxelspacing: Tuple[Real, Real],
) -> Dict[str, Real]:
    """Computes the clinical metrics for a single patient.

    Args:
        results: 2x (2,H,W), Predicted segmentation maps for both the 2- and 4-chamber views.The segmentation map for
            each view is a 3D stack of 2D segmentation maps for the ED and ES instants of the cardiac cycle.
        references: 2x (2,H,W), Reference segmentation maps for both the 2- and 4-chamber views. The segmentation map
            for each view is a 3D stack of 2D segmentation maps for the ED and ES instants of the cardiac cycle.
        a2c_voxelspacing: Size (in mm) of the 2-chamber view's voxels along each (height, width) dimension.
        a4c_voxelspacing: Size (in mm) of the 4-chamber view's voxels along each (height, width) dimension.

    Returns:
        Mapping between the clinical metrics' names and their value for the patient.
    """
    # Extract left ventricle masks from the data
    results_masks = [np.isin(result, Label.LV.value) for result in results]
    references_masks = [np.isin(reference, Label.LV.value) for reference in references]

    # Compute clinical metrics on groundtruth
    gt_lv_edv, gt_lv_esv = compute_left_ventricle_volumes(
        *references_masks[0], a2c_voxelspacing, *references_masks[1], a4c_voxelspacing
    )
    gt_lv_ef = (gt_lv_edv - gt_lv_esv) / gt_lv_edv

    # Compute clinical metrics on prediction
    lv_edv, lv_esv = compute_left_ventricle_volumes(
        *results_masks[0], a2c_voxelspacing, *results_masks[1], a4c_voxelspacing
    )
    lv_ef = (lv_edv - lv_esv) / lv_edv

    # Compute error between prediction and groundtruth
    metrics = {
        "lv_edv_error": 100 * abs(lv_edv - gt_lv_edv) / gt_lv_edv,
        "lv_esv_error": 100 * abs(lv_esv - gt_lv_esv) / gt_lv_esv,
        "lv_ef_error": 100 * abs(lv_ef - gt_lv_ef),
    }
    return metrics
