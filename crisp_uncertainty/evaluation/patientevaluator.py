from typing import Callable, Dict, List, Tuple

import numpy as np
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.evaluator import Evaluator
from crisp_uncertainty.utils.numpy import prob_to_categorical


class PatientEvaluator(Evaluator):
    """Generic class to evaluate predictions for each patient."""

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Computes the evaluation and returns metrics to be uploaded.

        Returns:
            Dict of average results, Dict of patient results
        """
        raise NotImplementedError

    @staticmethod
    def get_patient_confidences_and_accuracies(
        results: List[PatientResult], accuracy_fn: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes confidences and accuracies for all patients with samples.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.
            accuracy_fn: callable to evaluate the accuracy of the sample prediction

        Returns:
            average confidences and accuracies for each bin.
        """
        confidences = []
        accuracies = []
        for patient in results:
            for view, data in patient.views.items():
                for instant in range(data.pred.shape[0]):
                    pred = prob_to_categorical(data.pred[instant])
                    accuracy = accuracy_fn(pred, data.gt[instant])
                    # Flip uncertainties to get confidence scores
                    confidences.append(1 - data.frame_uncertainties[instant])
                    accuracies.append(accuracy)

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        return confidences, accuracies

    @staticmethod
    def get_pixel_confidences_and_accuracies(results: List[PatientResult]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes confidences and accuracies for all patients with respect pixels.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average confidences and accuracies for each bin.
        """
        confidences = []
        accuracies = []
        preds = []
        for patient in results:
            for view, data in patient.views.items():
                pred = prob_to_categorical(data.pred)
                correct_map = np.equal(pred, data.gt).astype(int)
                # Flip uncertainties to get confidence scores
                confidences.append(1 - data.uncertainty_map.flatten())
                accuracies.append(correct_map.flatten())
                preds.append(pred.flatten())

        confidences = np.concatenate(confidences)
        accuracies = np.concatenate(accuracies)
        preds = np.concatenate(preds)

        return confidences, accuracies, preds
