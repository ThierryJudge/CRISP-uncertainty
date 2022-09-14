from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from crisp_uncertainty.utils.numpy import prob_to_categorical
from medpy import metric
from vital.data.config import DataTag


class SegmentationMetrics(PatientEvaluator):
    """Segmentation metric evaluator.

    Args:
        labels: labels on which to compute metrics.
    """

    def __init__(self, labels: Tuple[DataTag, ...], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.labels = labels

        self.scores = {"dice": metric.dc}
        self.distances = {"hausdorff": metric.hd, "assd": metric.assd}

    def compute_binary_metrics(self, pred, gt, label_name, label_value, voxelspacing) -> Dict:
        """Computes binary segmentation metrics on one label.

        Args:
            pred: ndarray of prediction
            gt: ndarray of prediction
            label_name: string of the label
            label_value: int value for the label
            voxelspacing: ndarray of voxelspacing

        Returns:
            Dictionary of results.
        """
        metrics = {}
        pred_mask, gt_mask = np.isin(pred, label_value), np.isin(gt, label_value)

        # Compute the reconstruction accuracy metrics
        metrics.update(
            {f"{label_name}_{score}": score_fn(pred_mask, gt_mask) for score, score_fn in self.scores.items()}
        )

        # Compute the distance metrics (that require the images' voxelspacing)
        # only if the requested label is present in both result and reference
        if np.any(pred_mask) and np.any(gt_mask):
            metrics.update(
                {
                    f"{label_name}_{dist}": dist_fn(pred_mask, gt_mask, voxelspacing=voxelspacing)
                    for dist, dist_fn in self.distances.items()
                }
            )
        # Otherwise mark distances as NaN for this item
        else:
            metrics.update({f"{label_name}_{distance}": np.NaN for distance in self.distances})

        return metrics

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Call evaluation.

        Args:
            results: List of patient results on which to compute metrics

        Returns:
            average results to log
        """
        full_metrics = {}
        for patient in results:
            for view, data in patient.views.items():
                voxelspacing = data.voxelspacing[1:]
                for instant, i in data.instants.items():
                    pred = prob_to_categorical(data.pred[i])
                    gt = data.gt[i]

                    metrics = {}
                    if len(self.labels) > 2:
                        for label in self.labels:
                            metrics.update(self.compute_binary_metrics(pred, gt, str(label), label.value, voxelspacing))
                    else:
                        metrics.update(self.compute_binary_metrics(pred, gt, str(self.labels[1]), 1, voxelspacing))

                    # Compute average over all classes
                    for score in list(self.scores.keys()) + list(self.distances.keys()):
                        metrics[score] = np.mean(
                            [res for key, res in metrics.items() if score in key and "bg" not in key]
                        )

                    full_metrics[f"{patient.id}_{view}_{instant}"] = metrics

        full_metrics_df = pd.DataFrame(full_metrics)
        results = full_metrics_df.T.mean().to_dict()

        return results, full_metrics
