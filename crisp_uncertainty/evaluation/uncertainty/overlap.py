from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from crisp_uncertainty.utils.numpy import prob_to_categorical


class UncertaintyErrorOverlap(PatientEvaluator):
    """Evaluator for uncertainty error overlap.

    Args:
        uncertainty_threshold: threshold for the uncertainty to generate binary mask.
    """

    def __init__(self, uncertainty_threshold: float = 0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold

    @staticmethod
    def compute_overlap(error, uncertainty, uncertainty_threshold):
        """Computes overlap between error and uncertainty.

        Args:
            error: numpy binary array indicating error.
            uncertainty: numpy float array indicating uncertainty.
            uncertainty_threshold: Threshold to convert uncertianty to binary

        Returns:
            dice overlap.
        """
        uncertainty = (uncertainty > uncertainty_threshold).astype(int)
        intersection = np.logical_and(error, uncertainty)
        dice = 2.0 * intersection.sum() / (error.sum() + uncertainty.sum())

        return dice

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Computes uncertainty error overlap for all patients.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average uncertainty-error overlap.
        """
        metrics = {}
        error_sums = []
        for patient in results:
            for view, data in patient.views.items():
                for instant, i in data.instants.items():
                    pred = prob_to_categorical(data.pred[i])
                    error = 1 * ~np.equal(pred, data.gt[i])

                    error_sums.append(error.sum())

                    dice = self.compute_overlap(error, data.uncertainty_map[i], self.uncertainty_threshold)

                    metrics[f"{patient.id}_{view}_{instant}"] = {
                        "overlap": dice,
                        "uncertainty": data.frame_uncertainties[i],
                    }

        df = pd.DataFrame(metrics).T

        dice_list = np.array(df.overlap)
        mean = np.mean(dice_list)
        weighted_overlap = np.average(dice_list, weights=error_sums)
        dice_list[dice_list == 0] = np.nan
        non_zero_mean = np.nanmean(dice_list)

        # return {"non_zero_overlap": non_zero_mean, "overlap": mean, "weighted_overlap": weighted_overlap}, metrics
        return {"overlap": weighted_overlap, "overlap(nw)": mean}, metrics
