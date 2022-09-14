from typing import Dict, Tuple

import numpy as np
import torch
from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.systems.segmentation import SegmentationComputationMixin
from vital.utils.decorators import auto_move_data
from vital.utils.format.native import prefix


class LearningConfidence(UncertaintyMapEvaluationSystem, SegmentationComputationMixin):
    """Learning confidence uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, budget: float = 0.3, baseline: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("budget", "baseline")
        # Training dice requires activation function to be computed separately
        self._training_dice = DifferentiableDiceCoefficient(
            include_background=False, reduction="none", apply_activation=False
        )
        self.lmbda = 0.1

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        y, _ = self.module(*args, **kwargs)
        return y

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return f"LCE-{self.module.__class__.__name__}"

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        y_hat, confidence = self.module(x)
        binary = y_hat.shape[1] == 1

        if binary:
            y_hat = torch.sigmoid(y_hat)
            y_onehot = torch.clone(y.unsqueeze(1))
        else:
            y_hat = F.softmax(y_hat, dim=1)
            y_onehot = to_onehot(y, len(self.hparams.data_params.labels))

        confidence = torch.sigmoid(confidence)

        # Randomly set half of the confidences to 1 (i.e. no hints)
        # b = torch.bernoulli(torch.Tensor(confidence.shape[0]).uniform_(0, 1)).to(self.device)[..., None, None, None]
        b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(self.device)
        conf = confidence * b + (1 - b)

        y_hat_new = y_hat * conf.expand_as(y_hat) + y_onehot * (1 - conf.expand_as(y_onehot))

        # Segmentation accuracy metrics
        ce = F.binary_cross_entropy(y_hat_new.squeeze(), y.float()) if binary else F.nll_loss(y_hat_new, y)
        dice_values = self._training_dice(y_hat_new, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.cross_entropy_weight * ce) + (self.dice_weight * (1 - mean_dice))
        confidence_loss = torch.mean(-torch.log(confidence))

        total_loss = loss + (self.lmbda * confidence_loss)

        if self.hparams.budget > confidence_loss.item():
            self.lmbda = self.lmbda / 1.01
        elif self.hparams.budget <= confidence_loss.item():
            self.lmbda = self.lmbda / 0.99

        # Format output
        logs = {
            "loss": total_loss,
            "ce": ce,
            "dice": mean_dice,
            **dices,
            "confidence_loss": confidence_loss,
            "segmentation_loss": loss,
        }
        result = prefix(logs, "train_")
        self.log_dict(result, **self.train_log_kwargs)

        if batch_idx == 0:
            y_hat = y_hat.round() if binary else y_hat.argmax(1)
            self.log_images(
                title="Training Sample",
                num_images=5,
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_hat.detach().cpu().squeeze().numpy(),
                    "conf": confidence.detach().cpu().squeeze().numpy(),
                },
            )

        result["loss"] = result["train_loss"]
        return result

    @auto_move_data
    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        y_hat, confidence = self.module(img)

        if y_hat.shape[1] == 1:
            y_hat = torch.sigmoid(y_hat)
        else:
            y_hat = F.softmax(y_hat, dim=1)

        uncertainty = 1 - torch.sigmoid(confidence)
        return y_hat.cpu().numpy(), uncertainty.cpu().numpy()
