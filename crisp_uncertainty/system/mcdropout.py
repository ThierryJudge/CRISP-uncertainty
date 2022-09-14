from typing import Tuple

import numpy as np
import scipy.stats
import torch
from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from crisp_uncertainty.utils.mcdropout import patch_module
from torch import Tensor
from torch.nn import functional as F
from vital.systems.segmentation import SegmentationComputationMixin


class McDropoutUncertainty(UncertaintyMapEvaluationSystem, SegmentationComputationMixin):
    """MC Dropout system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, iterations: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("iterations")

        # Keep dropout at test time.
        self.module = patch_module(self.module)

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return f"McDropout-{self.module.__class__.__name__}_{int(self.module.dropout*100)}"

    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        logits = [self(img) for _ in range(self.hparams.iterations)]

        if logits[0].shape[1] == 1:
            probs = [torch.sigmoid(logits[i]).detach() for i in range(self.hparams.iterations)]
            probs = torch.stack(probs, dim=-1).cpu().numpy()
            y_hat = probs.mean(-1)
            # uncertainty_map = scipy.stats.entropy(probs, axis=-1)
            # uncertainty_map = 1 - (uncertainty_map - np.min(uncertainty_map))/np.ptp(uncertainty_map)
            y_hat_prime = np.concatenate([y_hat, 1 - y_hat], axis=1)
            uncertainty_map = scipy.stats.entropy(y_hat_prime, axis=1)
        else:
            probs = [F.softmax(logits[i], dim=1).detach() for i in range(self.hparams.iterations)]
            y_hat = torch.stack(probs, dim=-1).mean(-1).cpu().numpy()
            uncertainty_map = scipy.stats.entropy(y_hat, axis=1)

        return y_hat, uncertainty_map
