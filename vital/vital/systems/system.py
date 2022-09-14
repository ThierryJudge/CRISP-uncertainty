import sys
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchinfo import summary

from vital.data.config import DataParameters


class VitalSystem(pl.LightningModule, ABC):
    """Top-level abstract system from which to inherit.

    Implementations of behaviors related to each phase of training (e.g. data preparation, training, evaluation) are
    made through mixins for this class.

    Implements useful generic utilities and boilerplate Lighting code:
        - CLI for generic arguments
    """

    def __init__(self, data_params: DataParameters, lr: float, weight_decay: float, **kwargs):
        """Saves the parameters from all the model's childs and mixins in `hparams`.

        Args:
            data_params: Parameters related to the data necessary to initialize the model.
            lr: Learning rate for the Adam optimizer
            weight_decay: Weight decay for the Adam optimizer
            **kwargs: Dictionary of arguments to save as the model's `hparams`.
        """
        super().__init__()
        # Collection of hyperparameters configuring the system
        self.save_hyperparameters(ignore='module')

        # By default, assumes the provided data shape is in channel-first format
        self.example_input_array = torch.randn((2, *self.hparams.data_params.in_shape))

    def on_pretrain_routine_start(self) -> None:  # noqa: D102
        self.log_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    @property
    def log_dir(self) -> Path:
        """Returns the root directory where test logs get saved."""
        return Path(self.trainer.log_dir) if self.trainer.log_dir else Path.cwd()

    def summarize(self, mode: Optional[str] = "top", max_depth: Optional[int] = None) -> Optional[ModelSummary]:
        """Adds saving a Keras-style summary of the model to the base PL summary routine.

        The Keras-style summary is saved to a ``summary.txt`` file, inside the output directory.

        Notes:
            - Requires the ``example_input_array`` property to be set for the module.
            - Will not be printed if PL weights' summary was disabled (``mode == None``). This is done to avoid possible
              device incompatibilities in clusters.
        """
        if mode is not None:
            model_summary = summary(
                self,
                input_data=self.example_input_array,
                col_names=["input_size", "output_size", "kernel_size", "num_params"],
                depth=sys.maxsize,
                device=self.device,
                verbose=0,
            )
            (self.log_dir / "summary.txt").write_text(str(model_summary), encoding="utf-8")
        return super().summarize(mode)

    def configure_optimizers(self) -> Optimizer:  # noqa: D102
        # Todo move lr and weight decay to optim config.
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def log_images(
        self, title: str, num_images: int, axes_content: Dict[str, np.ndarray], info: Optional[List[str]] = None
    ):
        """Log images to Logger if it is a TensorBoardLogger or CometLogger.

        Args:
            title: Name of the figure.
            num_images: Number of images to log.
            axes_content: Mapping of axis name and image.
            info: Additional info to be appended to title for each image.
        """
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            if info is not None:
                name = f"{title}_{info[i]}_{i}"
            else:
                name = f"{title}_{i}"
            plt.suptitle(name)
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):
                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)

            if isinstance(self.trainer.logger, TensorBoardLogger):
                self.trainer.logger.experiment.add_figure("{}_{}".format(title, i), fig, self.current_epoch)
            if isinstance(self.trainer.logger, CometLogger):
                self.trainer.logger.experiment.log_figure("{}_{}".format(title, i), fig, step=self.current_epoch)

            plt.close()


class SystemComputationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the training/validation/testing phases."""

    def __init__(self, train_log_kwargs: dict, val_log_kwargs: dict, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters(ignore='module')

        self.train_log_kwargs = train_log_kwargs
        self.val_log_kwargs = val_log_kwargs

    def training_step(self, *args, **kwargs) -> Union[Tensor, Dict[Union[Literal["loss"], Any], Any]]:  # noqa: D102
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):  # noqa: D102
        pass


class SystemEvaluationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the evaluation phase."""

    def test_step(self, *args, **kwargs):  # noqa: D102
        pass

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """Runs at the end of a test epoch with the output of all test steps.

        It can be used to export results using custom loggers, while not returning any metrics to display in the
        progress bar (as Lightning usually expects).
        """
        pass
