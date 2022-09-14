from typing import Dict, List, Tuple

import numpy as np
from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from crisp_uncertainty.utils.numpy import prob_to_categorical
from matplotlib import pyplot as plt
from vital.data.camus.config import Label


class Distribution(PatientEvaluator):
    """Generates distribution figures for uncertainties."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results: List[PatientResult]) -> Tuple[Dict, Dict]:
        """Generates distributions for pixels, foreground pixels and samples.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            No metrics.
        """
        sample_uncertainties = []
        pixel_uncertainties = []
        fg_pixel_uncertainties = []
        for patient in results:
            for view, data in patient.views.items():
                mask = prob_to_categorical(data.pred) != Label.BG.value

                fg_pixel_uncertainties.append(data.uncertainty_map[mask].flatten())
                pixel_uncertainties.append(data.uncertainty_map.flatten())
                sample_uncertainties.append(data.frame_uncertainties)

        pixel_uncertainties = np.concatenate(pixel_uncertainties)
        fg_pixel_uncertainties = np.concatenate(fg_pixel_uncertainties)
        sample_uncertainties = np.concatenate(sample_uncertainties)

        num_bins = 20
        plt.figure()
        plt.title("Pixel Uncertainty Distribution")
        n, bins, patches = plt.hist(pixel_uncertainties, num_bins, facecolor="blue")
        plt.savefig(self.upload_dir / "pixel_uncertainty_distribution.png", dpi=100)
        plt.close()

        plt.figure()
        plt.title("ForegroundPixel Uncertainty Distribution")
        n, bins, patches = plt.hist(fg_pixel_uncertainties, num_bins, facecolor="blue")
        plt.savefig(self.upload_dir / "fg_pixel_uncertainty_distribution.png", dpi=100)
        plt.close()

        plt.figure()
        plt.title("Sample Uncertainty Distribution")
        n, bins, patches = plt.hist(sample_uncertainties, num_bins, facecolor="blue")
        plt.savefig(self.upload_dir / "sample_uncertainty_distribution.png", dpi=100)
        plt.close()

        return {}, {}
