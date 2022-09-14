from typing import Dict

from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from vital.data.config import Tags
from vital.systems.computation import TrainValComputationMixin


class ClassificationComputationMixin(TrainValComputationMixin):
    """Mixin for classification train/val step.

    Implements generic classification train/val step and inference, assuming the following conditions:
        - the ``nn.Module`` used returns as single output the raw, unnormalized scores for each class.
    """

    # Fields to initialize in implementation of ``VitalSystem``
    #: Network called by ``ClassificationComputationMixin`` for test-time inference
    module: nn.Module

    def __init__(self, module: nn.Module, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            module: Network to train.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)

        self.module = module

    def forward(self, *args, **kwargs):  # noqa: D102
        return self.module(*args, **kwargs)

    def trainval_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        y_hat = self.module(x)

        # Loss and metrics
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        if self.is_val_step and batch_idx == 0:
            self.log_images(
                title="Sample",
                num_images=5,
                axes_content={"Image": x.cpu().squeeze().numpy()},
                info=[f"(Gt: {y[i].item()} - Pred: {y_hat.argmax(dim=1)[i].item()})" for i in range(5)],
            )

        # Format output
        return {"loss": loss, "accuracy": acc}
