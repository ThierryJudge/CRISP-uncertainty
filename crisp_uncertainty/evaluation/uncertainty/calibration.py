from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from matplotlib import pyplot as plt
from vital.data.camus.config import Label

from crisp_uncertainty.utils.numpy import prob_to_categorical


class Calibration(PatientEvaluator):
    """Abstract calibration evaluator.

    Args:
        nb_bins: number of bin for the calibration computation.
    """

    CALIBRATION_FILE_NAME: str = None

    def __init__(self, nb_bins: int = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_bins = nb_bins
        bin_boundaries = np.linspace(0, 1, nb_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def get_confidences_and_accuracies(self, results: List[PatientResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Computes confidences and accuracies for all patients with respect to desired target.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average confidences and accuracies for each bin.
        """
        raise NotImplementedError

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Computes calibration for all patients with respect to desired target.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            expected calibration error.
        """
        confidences, accuracies = self.get_confidences_and_accuracies(results)

        ece = np.zeros(1)
        bins_avg_conf = []
        bins_avg_acc = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bins_avg_conf.append(avg_confidence_in_bin)
                bins_avg_acc.append(accuracy_in_bin)

        np.save(self.upload_dir / f"{self.__class__.__name__}.npy", {"conf": bins_avg_conf, "acc": bins_avg_acc})

        plt.figure()
        plt.plot(bins_avg_conf, bins_avg_acc)
        plt.title(f"{self.__class__.__name__}")
        plt.savefig(self.upload_dir / f"{self.__class__.__name__}.png", dpi=100)
        plt.close()

        return {f"{self.__class__.__name__}_ece": float(ece)}, {}

    @classmethod
    def export_results(
        cls, experiment_names: List[str], data_dir: Path, num_rows: Optional[int] = None, num_cols: Optional[int] = None
    ):
        """Aggregates and exports results for evaluator.

        Args:
            experiment_names: List of experiment names.
            data_dir: Path to the downloaded data
            num_rows: Number of rows for subplots.
            num_cols: Number of columns for subplots.
        """
        plt.figure()
        plt.title(cls.__name__)
        plt.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
        plt.ylabel("Accuracy")
        plt.xlabel("Confidence")
        for i, exp in enumerate(experiment_names):
            data = np.load(data_dir / exp / cls.CALIBRATION_FILE_NAME, allow_pickle=True)[()]
            plt.plot(data["conf"], data["acc"], label=exp, marker="o")

        plt.legend()
        plt.savefig(data_dir / f"{cls.__name__}.png", dpi=100)
        plt.close()


class PixelCalibration(Calibration):
    """Calibration evaluation per pixel."""

    CALIBRATION_FILE_NAME = "PixelCalibration.npy"
    SAVED_FILES = [CALIBRATION_FILE_NAME]

    def get_confidences_and_accuracies(self, results: List[PatientResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Computes confidences and accuracies for all patients with respect pixels.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average confidences and accuracies for each bin.
        """
        confidences, accuracies, preds = self.get_pixel_confidences_and_accuracies(results)
        not_bg = preds != Label.BG.value
        confidences = confidences[not_bg]
        accuracies = accuracies[not_bg]

        return confidences, accuracies


class SampleCalibration(Calibration):
    """Calibration evaluation per sample.

    Args:
        accuracy_fn: callable to evaluate the accuracy of the sample prediction.
    """

    CALIBRATION_FILE_NAME = "SampleCalibration.npy"
    SAVED_FILES = [CALIBRATION_FILE_NAME]

    def __init__(self, accuracy_fn: Callable, *args, **kwargs):
        self.accuracy_fn = accuracy_fn
        super().__init__(*args, **kwargs)

    def get_confidences_and_accuracies(self, results: List[PatientResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Computes confidences and accuracies for all patients with samples.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average confidences and accuracies for each bin.
        """
        return self.get_patient_confidences_and_accuracies(results, self.accuracy_fn)


class PatientCalibration(PatientEvaluator):
    """Abstract calibration evaluator.

    Args:
        nb_bins: number of bin for the calibration computation.
    """

    CALIBRATION_FILE_NAME: str = None

    def __init__(self, nb_bins: int = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_bins = nb_bins
        bin_boundaries = np.linspace(0, 1, nb_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def ece(self, confidences, accuracies):
        ece = np.zeros(1)
        bins_avg_conf = []
        bins_avg_acc = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bins_avg_conf.append(avg_confidence_in_bin)
                bins_avg_acc.append(accuracy_in_bin)
        return ece

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Computes calibration for all patients with respect to desired target.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            expected calibration error.
        """
        metrics = {}

        for patient in results:
            for view, data in patient.views.items():
                for instant, i in data.instants.items():
                    pred = prob_to_categorical(data.pred[i])
                    correct_map = np.equal(pred, data.gt[i]).astype(int)
                    ece = self.ece(1 - data.uncertainty_map[i].flatten(), correct_map.flatten())

                    metrics[f"{patient.id}_{view}_{instant}"] = {"ece": ece}

        df = pd.DataFrame(metrics).T

        ece_list = np.array(df.ece)
        ece = np.mean(ece_list)

        return {f"sample_ece": float(ece)}, {}
