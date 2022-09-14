from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from crisp_uncertainty.evaluation.evaluator import Evaluator
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


class DatasetEvaluator(Evaluator):
    """Generic class to evaluate predictions for full dataset."""

    def __call__(self, patient_results: pd.DataFrame) -> Dict[str, float]:
        """Computes the evaluation and returns metrics to be uploaded.

        Args:
            patient_results: dataframe containing metrics for each sample for each patient.

        Returns:
            Dict of average results
        """
        raise NotImplementedError


class DiceErrorCorrelation(DatasetEvaluator):
    """Evaluator to compute dice between segmentation dice score and uncertainty error overlap."""

    DICE_OVERLAP_FILE = "DiceOverlap.npy"
    SAVED_FILES = [DICE_OVERLAP_FILE]

    def __call__(self, patient_results: pd.DataFrame) -> Dict[str, float]:
        """Computes the correlation.

        Args:
            patient_results: dataframe containing metrics for each sample for each patient.

        Returns:
            Pearson correlation score.
        """
        dices = np.array(patient_results.dice)
        overlaps = np.array(patient_results.overlap)
        mis = np.array(patient_results.mutual_info)
        overlap_corr, _ = pearsonr(dices, overlaps)
        mi_corr, _ = pearsonr(dices, mis)

        plt.figure()
        plt.scatter(dices, overlaps)
        plt.xlabel("Dice")
        plt.ylabel("U-E overlap")
        plt.savefig(self.upload_dir / "dice_overlap_correlation.png", dpi=100)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(self.upload_dir / "dice_overlap_correlation_full.png", dpi=100)
        plt.close()

        plt.figure()
        plt.scatter(dices, mis)
        plt.xlabel("Dice")
        plt.ylabel("U-E MI")
        plt.savefig(self.upload_dir / "dice_mi_correlation.png", dpi=100)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(self.upload_dir / "dice_mi_correlation_full.png", dpi=100)
        plt.close()

        np.save(self.upload_dir / self.DICE_OVERLAP_FILE, {"overlap": overlaps, "dice": dices, "mis": mis})

        return {"dice_overlap_correlation": overlap_corr, "dice_mi_correlation": mi_corr}

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

        for i, exp in enumerate(experiment_names):
            data = np.load(data_dir / exp / cls.DICE_OVERLAP_FILE, allow_pickle=True)[()]
            axes1[i].scatter(data["dice"], data["overlap"])
            axes1[i].set_xlabel("Dice")
            axes1[i].set_ylabel("E-U Overlap")
            axes1[i].set_title(exp)

        f1.tight_layout()
        f1.savefig(data_dir / "DiceOverlapCorrelation.png", dpi=100)
        plt.close()


class ThresholdedDice(DatasetEvaluator):
    """Evaluator to compute dice averages for multiple subsets of the dataset according to uncertainty."""

    def __call__(self, patient_results: pd.DataFrame) -> Dict[str, float]:
        """Computes the thresholded dice averages.

        Args:
            patient_results: dataframe containing metrics for each sample for each patient.

        Returns:
            No metrics.
        """
        uncertainties = np.array(patient_results.uncertainty)
        dices = np.array(patient_results.dice)

        thresh_dices = []
        for i in np.linspace(0, np.max(uncertainties), 10):
            thresh_dices.append(np.mean(dices[uncertainties > i]))

        plt.figure()
        plt.plot(np.linspace(0, np.max(uncertainties), 10), thresh_dices)
        plt.xlabel("Uncertainty threshold")
        plt.ylabel("Dice")
        plt.savefig(self.upload_dir / "threshold_dice.png", dpi=100)
        plt.close()

        return {}


class ThresholdErrorOverlap(DatasetEvaluator):
    """Evaluator to compute dice between segmentation dice score and uncertainty error overlap."""

    def __init__(self, dice_threshold: float = 0.75, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice_threshold = dice_threshold
        self.dice_threshold2 = 0.6

    def __call__(self, patient_results: pd.DataFrame) -> Dict[str, float]:
        dices = np.array(patient_results.dice)

        overlaps = np.array(patient_results.overlap)
        overlaps_thresh = overlaps[dices < self.dice_threshold]
        overlaps_50 = overlaps[dices < self.dice_threshold2]

        mis = np.array(patient_results.mutual_info)
        mis_thresh = mis[dices < self.dice_threshold]
        mis_50 = mis[dices < self.dice_threshold2]

        overlap_mean = np.mean(overlaps_thresh)
        mi_mean = np.mean(mis_thresh)
        overlap_mean_50 = np.mean(overlaps_50)
        mi_mean_50 = np.mean(mis_50)

        quartile = dices < np.percentile(dices, 25)
        overlap_quartile = np.mean(overlaps[quartile])
        mi_quartile = np.mean(mis[quartile])

        return {f"thresh.({self.dice_threshold})_overlap": overlap_mean,
                f"thresh.({self.dice_threshold})_mutual_info": mi_mean,
                f"thresh.({self.dice_threshold2})_overlap": overlap_mean_50,
                f"thresh.({self.dice_threshold2})_mutual_info": mi_mean_50,
                f"thresh.(quartile)_overlap": overlap_quartile,
                f"thresh.(quartile)_mutual_info": mi_quartile
                }
