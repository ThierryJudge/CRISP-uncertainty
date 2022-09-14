from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributions as distributions
from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.systems.segmentation import SegmentationComputationMixin
from vital.utils.decorators import auto_move_data


class AleatoricUncertainty(UncertaintyMapEvaluationSystem, SegmentationComputationMixin):
    """Aleatoric uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, iterations: int = 100, is_log_sigma: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("iterations", "is_log_sigma")
        self.iterations = iterations
        self.is_log_sigma = is_log_sigma
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none", apply_activation=False)

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        return self.module(*args, **kwargs)

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return f"Aleatoric-{self.module.__class__.__name__}"

    def trainval_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        logits, sigma = self(x)  # (N, C, H, W), (N, C, H, W)
        binary = logits.shape[1] == 1

        if self.is_log_sigma:
            distribution = distributions.Normal(logits, torch.exp(sigma))
        else:
            distribution = distributions.Normal(logits, sigma + 1e-8)

        x_hat = distribution.rsample((self.iterations,))

        if binary:
            mc_expectation = torch.sigmoid(x_hat).mean(dim=0)
            ce = F.binary_cross_entropy(mc_expectation.squeeze(), y.float())
        else:
            mc_expectation = F.softmax(x_hat, dim=2).mean(dim=0)
            log_probs = mc_expectation.log()
            ce = F.nll_loss(log_probs, y)

        dice_values = self._dice(mc_expectation, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.cross_entropy_weight * ce) + (self.dice_weight * (1 - mean_dice))

        if self.is_val_step and batch_idx == 0:
            if binary:
                y_hat = torch.sigmoid(logits).round()
                sigma_pred = sigma
            else:
                y_hat = logits.argmax(dim=1)
                prediction_onehot = to_onehot(y_hat, num_classes=len(self.hparams.data_params.labels)).type(torch.bool)
                sigma_pred = torch.where(prediction_onehot, sigma, sigma * 0).sum(dim=1)

            self.log_images(
                title="Sample",
                num_images=5,
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_hat.detach().cpu().squeeze().numpy(),
                    "Sigma": sigma_pred.detach().cpu().squeeze().numpy(),
                },
            )

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        logits, sigma = self(img)

        if logits.shape[1] == 1:
            y_hat = torch.sigmoid(logits)
        else:
            y_hat = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            prediction_onehot = to_onehot(pred, num_classes=len(self.hparams.data_params.labels)).type(torch.bool)
            sigma = torch.where(prediction_onehot, sigma, sigma * 0).sum(dim=1)

        return y_hat.cpu().numpy(), sigma.cpu().numpy().squeeze()
