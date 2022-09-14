from typing import List

import numpy as np
from medpy.metric import dc
from vital.data.camus.config import Label
from vital.data.config import DataTag


class Dice:
    """Dice computation with Medpy package.

    Args:
        labels: List of labels for which to compute the dice.
        exclude_bg: If true, background dice is not considered.
    """

    def __init__(self, labels: List[DataTag], exclude_bg: bool = True):
        self.labels = labels
        self.exclude_bg = exclude_bg

    def __call__(self, pred: np.ndarray, target: np.ndarray):
        """Compute dice for one sample.

        Args:
            pred: prediction array in categorical form (H, W)
            target: target array in categorical form (H, W)

        Returns:
            mean dice
        """
        dices = []
        if len(self.labels) > 2:
            for label in self.labels:
                if self.exclude_bg and label.value == Label.BG.value:
                    pass
                else:
                    pred_mask, gt_mask = np.isin(pred, label.value), np.isin(target, label.value)
                    dices.append(dc(pred_mask, gt_mask))
            return np.array(dices).mean()
        else:
            return dc(pred.squeeze(), target.squeeze())


class Accuracy:
    """Accuracy computation."""

    def __call__(self, pred: np.ndarray, target: np.ndarray):
        """Compute accuracy for one sample.

        Args:
            pred: prediction array in categorical form (H, W)
            target: target array in categorical form (H, W)

        Returns:
            mean dice
        """
        return np.equal(pred, target).mean()
