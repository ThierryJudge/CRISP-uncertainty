from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


class Correlation(PatientEvaluator):
    """Correlation between dice and uncertainty evaluator.

    Args:
        accuracy_fn: callable to evaluate the accuracy of the sample prediction.
    """

    CORRELATION_FILE_NAME = "Correlation.npy"
    SAVED_FILES = [CORRELATION_FILE_NAME]

    def __init__(self, accuracy_fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_fn = accuracy_fn

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict, Dict]:
        """Correlation between dice and uncertainty for all samples with respect to accuracy function.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            Pearson correlation score.
        """
        confidences, accuracies = self.get_patient_confidences_and_accuracies(results, self.accuracy_fn)
        uncertainty = -(confidences - 1)

        plt.figure()
        plt.scatter(uncertainty, accuracies)
        plt.xlabel("Uncertainty")
        plt.ylabel("Accuracy measure")
        plt.savefig(self.upload_dir / "Correlation.png", dpi=100)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(self.upload_dir / "Correlation_full.png", dpi=100)
        plt.close()

        np.save(self.upload_dir / self.CORRELATION_FILE_NAME, {"conf": confidences, "acc": accuracies})

        corr, _ = pearsonr(uncertainty[np.isfinite(uncertainty)], accuracies[np.isfinite(uncertainty)])

        return {self.__class__.__name__: float(np.abs(corr))}, {}

    @classmethod
    def export_results(
            cls, experiment_names: List[str], data_dir: Path, num_rows: Optional[int] = None,
            num_cols: Optional[int] = None
    ):
        """Aggregates and exports results for evaluator.

        Args:
            experiment_names: List of experiment names.
            data_dir: Path to the downloaded data
            num_rows: Number of rows for subplots.
            num_cols: Number of columns for subplots.
        """
        num_rows = num_rows or 1
        num_cols = num_cols or len(experiment_names)
        f1, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
        f1.set_figheight(9)
        f1.set_figwidth(16)
        axes1 = axes.ravel()

        f2, axes = plt.subplots(num_rows, num_cols)
        f2.set_figheight(9)
        f2.set_figwidth(16)
        axes2 = axes.ravel()

        f3 = plt.figure()
        ax3 = plt.gca()
        markers = ['o', '+', 'x', '*', '.', 'X']

        for i, exp in enumerate(experiment_names):
            data = np.load(data_dir / exp / cls.CORRELATION_FILE_NAME, allow_pickle=True)[()]
            axes1[i].scatter(data["conf"], data["acc"])
            axes1[i].set_xlabel("Confidence")
            axes1[i].set_ylabel("Accuracy measure")
            axes1[i].set_title(exp)

            conf = data['conf']

            # if conf.min() < 0 or conf.max() > 1:
            min = conf.min()
            max = conf.max()
            conf = (conf - min) / (max - min)

            # b, m = np.polyfit(conf, data["acc"], 1)
            # axes2[i].plot(conf, b + m * data["conf"], '-')
            axes2[i].scatter(conf, data["acc"])
            axes2[i].set_xlabel("Confidence")
            axes2[i].set_ylabel("Accuracy measure")
            axes2[i].set_title(exp)
            axes2[i].set_xlim([0, 1])
            axes2[i].set_ylim([0, 1])

            ax3.scatter(conf, data["acc"], label=exp, marker=markers[i])

        f3.legend()

        f1.tight_layout()
        f2.tight_layout()
        f3.tight_layout()
        f1.savefig(data_dir / "Correlation.png", dpi=100)
        f2.savefig(data_dir / "Correlation_full.png", dpi=100)
        f3.savefig(data_dir / "Correlation_all.png", dpi=100)
        plt.close()
