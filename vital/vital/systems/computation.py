from abc import ABC
from typing import Dict

from torch import Tensor

from vital.systems.system import SystemComputationMixin
from vital.utils.format.native import prefix


class TrainValComputationMixin(SystemComputationMixin, ABC):
    """Abstract mixin for generic supervised train/val step.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def __init__(self, train_log_kwargs: dict, val_log_kwargs: dict, **kwargs):
        super().__init__(train_log_kwargs, val_log_kwargs, **kwargs)
        self.is_val_step = False

    def trainval_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for both training and validation loops, assuming the behavior should be the same.

        For models where the behavior in training and validation is different, then override ``training_step`` and
        ``validation_step`` directly (in which case ``trainval_step`` doesn't need to be implemented).

        Returns:
            Mapping between metric names and their values. It must contain at least a ``'loss'``, as that is the value
            optimized in training and monitored by callbacks during validation.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self.trainval_step(*args, **kwargs), "train_")
        self.log_dict(result, **self.train_log_kwargs)
        # Add reference to 'train_loss' under 'loss' keyword, requested by PL to know which metric to optimize
        result["loss"] = result["train_loss"]
        return result

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        self.is_val_step = True
        result = prefix(self.trainval_step(*args, **kwargs), "val_")
        self.is_val_step = False
        result.update({"early_stop_on": result["val_loss"]})
        self.log_dict(result, **self.val_log_kwargs)
        return result
