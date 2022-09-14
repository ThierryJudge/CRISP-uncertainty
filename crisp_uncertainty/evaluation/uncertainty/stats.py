from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from crisp_uncertainty.utils.numpy import prob_to_categorical


class Stats(PatientEvaluator):
    """Evaluator for uncertainty error overlap.

    Args:
        uncertainty_threshold: threshold for the uncertainty to generate binary mask.
    """

    def __init__(self, uncertainty_threshold: float = 0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Computes uncertainty error overlap for all patients.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average uncertainty-error overlap.
        """
        uncertainties = []
        errors = []
        for patient in results:
            for view, data in patient.views.items():
                for instant, i in data.instants.items():
                    pred = prob_to_categorical(data.pred[i])
                    error = 1 * ~np.equal(pred, data.gt[i])

                    uncertainty = (data.uncertainty_map[i] > self.uncertainty_threshold).astype(int)
                    errors.append(error.flatten())
                    uncertainties.append(uncertainty.flatten())

        uncertainties = np.array(uncertainties).flatten()
        errors = np.array(errors).flatten()
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(uncertainties == 1, errors == 1))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(uncertainties == 0, errors == 0))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(uncertainties == 1, errors == 0))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(uncertainties == 0, errors == 1))

        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)

        return {"FPR": FPR, "FNR": FNR}, {}
