from typing import Tuple

import numpy as np
import scipy.stats
import torch
from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from torch import Tensor
from torch.nn import functional as F
from vital.systems.segmentation import SegmentationComputationMixin


class UncertaintyFunction:
    """Abstract class for uncertainty functions for softmax uncertainty evaluation.

    Args:
        flip: If True, values are flipped because smaller values indicate higher uncertainty.
    """

    def __init__(self, flip: bool):
        self.flip = flip

    def __call__(self, x: np.ndarray):
        """Calls the uncertainty computation and flips if needed.

        Args:
            x: input segmentation map.

        Returns:
            Uncertainty map.
        """
        uncertainty = self.uncertainty(x)
        if self.flip:
            uncertainty = 1 - uncertainty
        return uncertainty

    def uncertainty(self, x: np.ndarray):
        """Computes the uncertainty function for a given sample.

        Args:
            x: input segmentation map.

        Returns:
            Uncertainty or certainty map.
        """
        raise NotImplementedError


class Max(UncertaintyFunction):
    """Computes uncertainty using the maximum along the softmax axis."""

    def __init__(self):
        super().__init__(flip=True)

    def uncertainty(self, x: np.ndarray):  # noqa D102
        if x.shape[1] == 1:
            return np.abs(x - 0.5) / 0.5  # If binary, center on 0, take absolute value and normalize.
        return np.max(x, axis=1)


class Entropy(UncertaintyFunction):
    """Computes uncertainty using the entropy along the softmax axis."""

    def __init__(self):
        super().__init__(flip=False)

    def uncertainty(self, x: np.ndarray):  # noqa D102
        if x.shape[1] == 1:
            x = np.concatenate([x, 1 - x], axis=1)
        return scipy.stats.entropy(x, axis=1)


class SoftmaxUncertainty(UncertaintyMapEvaluationSystem, SegmentationComputationMixin):
    """System for evaluating uncertainty using the Softmax probabilities.

    Args:
        mode: Name of the uncertainty function to use.
        *args: Positional arguments to pass to the parent's constructor.
        **kwargs: Keyword arguments to pass to the parent's constructor.
    """

    available_modes = {
        "max": Max(),
        "entropy": Entropy(),
    }

    def __init__(self, mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("mode")
        self.mode = self.available_modes[mode]

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return f"softmax-{self.hparams.mode}"

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

        uncertainty_map = self.mode(y_hat)

        return y_hat, uncertainty_map
