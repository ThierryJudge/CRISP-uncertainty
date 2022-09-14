from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from crisp_uncertainty.utils.numpy import prob_to_categorical


class UncertaintyErrorMutualInfo(PatientEvaluator):
    """Evaluator for uncertainty error overlap.

    Args:
        uncertainty_threshold: threshold for the uncertainty to generate binary mask.
    """

    def __init__(self, bins=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins

    @staticmethod
    def compute_mi(error, uncertainty):
        """Computes mutual information between error and uncertainty.

        Args:
            error: numpy binary array indicating error.
            uncertainty: numpy float array indicating uncertainty.

        Returns:
            mutual_information
        """
        hist_2d, x_edges, y_edges = np.histogram2d(error.ravel(), uncertainty.ravel())

        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)  # marginal for x over y
        py = np.sum(pxy, axis=0)  # marginal for y over x
        px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

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

                    mi = self.compute_mi(error, data.uncertainty_map[i])

                    metrics[f"{patient.id}_{view}_{instant}"] = {
                        "mutual_info": mi,
                    }

        df = pd.DataFrame(metrics).T

        mi_list = np.array(df.mutual_info)
        mean = np.mean(mi_list)
        weighted_mi = np.average(mi_list, weights=error_sums)

        # return {"mutual_info": mean, "weighted_mutual_info": weighted_mi}, metrics
        return {"mutual_info": weighted_mi, "mutual_info(mw)": mean}, metrics
