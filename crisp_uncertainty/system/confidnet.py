from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from pytorch_lightning.core.memory import ModelSummary
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from vital.data.config import Tags
from vital.systems.segmentation import SegmentationComputationMixin
from vital.utils.decorators import auto_move_data


class ConfidNet(UncertaintyMapEvaluationSystem, SegmentationComputationMixin):
    """Learning confidence uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_layers()  # Freeze all layers except confidence network.
        self.loss = nn.MSELoss()

    def freeze_layers(self):
        """Freezes layers for fine-tuning."""
        print("Freezing every layer except uncertainty")
        for param in self.module.named_parameters():
            if "conf" in param[0]:
                # print(param[0], "kept to training")
                continue
            param[1].requires_grad = False

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:  # noqa: D102
        pass

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return f"ConfidNet-{self.module.__class__.__name__}"

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        y, _ = self.module(*args, **kwargs)
        return y

    def trainval_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        y_hat, confidence = self.module(x)  # (N, C, H, W), (N, 1, H, W)
        binary = y_hat.shape[1] == 1

        if binary:
            y_hat = torch.sigmoid(y_hat)
            target_confidence = y_hat
        else:
            y_hat = F.softmax(y_hat, dim=1)
            y_onehot = to_onehot(y, len(self.hparams.data_params.labels))
            target_confidence = (y_hat * y_onehot).sum(dim=1)  # Get value of probability attributed to GT class.

        confidence = torch.sigmoid(confidence)

        loss = self.loss(confidence.squeeze(), target_confidence.squeeze())
        mae = F.l1_loss(confidence.squeeze(), target_confidence.squeeze(), reduction="mean")

        logs = {"loss": loss, "mae": mae}

        if batch_idx == 0:
            if binary:
                y_hat = y_hat.round()
                tcp = torch.abs(target_confidence - 0.5) / 0.5
                conf = torch.abs(confidence - 0.5) / 0.5
            else:
                y_hat.argmax(1)
                tcp = target_confidence
                conf = confidence
            y_hat = y_hat.argmax(1) if y_hat.shape[1] > 1 else torch.sigmoid(y_hat).round()

            self.log_images(
                title=f"{'Validation' if self.is_val_step else 'Training'} Sample",
                num_images=5,
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_hat.detach().cpu().squeeze().numpy(),
                    "Conf.": conf.detach().cpu().squeeze().numpy(),
                    "TCP": tcp.detach().cpu().squeeze().numpy(),
                },
            )

        return logs

    @auto_move_data
    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        y_hat, confidence = self.module(img)
        confidence = torch.sigmoid(confidence)
        if y_hat.shape[1] == 1:
            y_hat = torch.sigmoid(y_hat)
            # If binary, confidence is sigmoid probability
            confidence = torch.abs(confidence - 0.5) / 0.5
        else:
            y_hat = F.softmax(y_hat, dim=1)

        confidence = 1 - confidence

        return y_hat.cpu().numpy(), confidence.cpu().numpy()
