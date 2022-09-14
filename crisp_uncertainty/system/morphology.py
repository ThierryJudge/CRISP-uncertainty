from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.morphology import dilation, erosion
from torch import Tensor
from torch.nn import functional as F
from vital.systems.segmentation import SegmentationComputationMixin

from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from crisp_uncertainty.utils.numpy import prob_to_categorical


class MorphologyUncertainty(UncertaintyMapEvaluationSystem, SegmentationComputationMixin):
    """System for evaluating uncertainty using the Softmax probabilities.

    Args:
        mode: Name of the uncertainty function to use.
        *args: Positional arguments to pass to the parent's constructor.
        **kwargs: Keyword arguments to pass to the parent's constructor.
    """

    def __init__(self, thickness: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('thickness')

    def get_name(self):
        return f"morphology-{self.module.__class__.__name__}-{self.hparams.thickness}_thick"

    @staticmethod
    def compute_morph_unc(pred: np.ndarray, thickness: int):

        dilatation_mask = np.copy(pred)
        erosion_mask = np.copy(pred)

        prev_erosion = np.copy(erosion_mask)
        prev_dilatation = np.copy(dilatation_mask)

        uncertainty_map = np.zeros_like(pred).squeeze()

        for j in range(thickness):
            dilatation_mask = dilation(dilatation_mask, selem=np.ones((3, 3)))
            erosion_mask = erosion(erosion_mask, selem=np.ones((3, 3)))

            erosion_edges = prev_erosion ^ erosion_mask
            dilatation_edges = prev_dilatation ^ dilatation_mask

            prev_erosion = np.copy(erosion_mask)
            prev_dilatation = np.copy(dilatation_mask)

            uncertainty_map = uncertainty_map + (1 + -j / thickness) * (erosion_edges.clip(max=1) + dilatation_edges.clip(max=1)).astype(float)

        return uncertainty_map

    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        y_hat = self(img)
        if y_hat.shape[1] == 1:
            y_hat = torch.sigmoid(y_hat).detach().cpu().numpy()
        else:
            y_hat = F.softmax(y_hat, dim=1).detach().cpu().numpy()

        uncertainty_map = []

        for i in range(y_hat.shape[0]):
            pred = prob_to_categorical(y_hat[i])

            unc = self.compute_morph_unc(pred, self.hparams.thickness)

            uncertainty_map.append(unc)

        return y_hat, np.array(uncertainty_map)
