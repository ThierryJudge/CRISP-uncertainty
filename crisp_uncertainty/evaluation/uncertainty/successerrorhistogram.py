from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from matplotlib import pyplot as plt
from vital.data.camus.config import Label


class SuccessErrorHist(PatientEvaluator):
    """Abstract calibration evaluator.

    Args:
        nb_bins: number of bin for the calibration computation.
    """

    HIST_FILE_NAME: str = "SuccessErrorHist.npy"
    SAVED_FILES = [HIST_FILE_NAME]

    def __init__(self, nb_bins: int = 21, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_bins = nb_bins

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Computes calibration for all patients with respect to desired target.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            expected calibration error.
        """
        conf, acc, pred = self.get_pixel_confidences_and_accuracies(results)

        not_bg = pred != Label.BG.value
        conf = conf[not_bg]
        acc = acc[not_bg]

        plt.figure()
        plt.hist(
            conf[np.where(acc == 1)],
            bins=np.linspace(0, 1, num=self.nb_bins),
            density=True,
            color="green",
            label="Successes",
        )
        plt.hist(
            conf[np.where(acc == 0)],
            bins=np.linspace(0, 1, num=self.nb_bins),
            density=True,
            alpha=0.5,
            color="red",
            label="Errors",
        )
        plt.xlabel("Confidence")
        plt.ylabel("Relative density")
        plt.xlim(left=0, right=1)
        plt.legend()
        plt.savefig(self.upload_dir / f"{self.__class__.__name__}.png", dpi=100)
        plt.close()

        np.save(self.upload_dir / self.HIST_FILE_NAME, {"conf": conf, "acc": acc})

        return {}, {}

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
        num_rows = num_rows or 1
        num_cols = num_cols or len(experiment_names)
        f, axes = plt.subplots(num_rows, num_cols)
        f.set_figheight(9)
        f.set_figwidth(16)
        axes = axes.ravel()

        for i, exp in enumerate(experiment_names):
            data = np.load(data_dir / exp / cls.HIST_FILE_NAME, allow_pickle=True)[()]
            axes[i].set_title(exp)
            axes[i].hist(
                data["conf"][np.where(data["acc"] == 1)],
                bins=np.linspace(0, 1, num=21),
                density=True,
                color="green",
                label="Successes",
            )
            axes[i].hist(
                data["conf"][np.where(data["acc"] == 0)],
                bins=np.linspace(0, 1, num=21),
                density=True,
                alpha=0.5,
                color="red",
                label="Errors",
            )
            axes[i].set_xlabel("Confidence")
            axes[i].set_ylabel("Relative density")
            axes[i].set_xlim(left=0, right=1)
            axes[i].legend()

        f.savefig(data_dir / f"{cls.__name__}.png", dpi=100)
        plt.close()