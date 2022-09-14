from torch import Tensor, nn

from vital.metrics.train.functional import differentiable_dice_score


class DifferentiableDiceCoefficient(nn.Module):
    """Computes a differentiable version of the dice coefficient."""

    def __init__(
        self,
        include_background: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = "elementwise_mean",
        apply_activation: bool = True,
    ):
        """Initializes class instance.

        Args:
            include_background: Whether to also compute dice for the background.
            nan_score: Score to return, if a NaN occurs during computation (denom zero).
            no_fg_score: Score to return, if no foreground pixel was found in target.
            reduction: Method for reducing metric score over labels.
                Available reduction methods:
                - ``'elementwise_mean'``: takes the mean (default)
                - ``'none'``: no reduction will be applied
            apply_activation: when True, softmax or sigmoid is applied to input.
        """
        super().__init__()
        self.include_background = include_background
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        assert reduction in ("elementwise_mean", "none")
        self.reduction = reduction
        self.apply_activation = apply_activation

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Actual metric calculation.

        Args:
            input: (N, C, H, W), Raw, unnormalized (or normalized apply_activation == False) if scores for each class.
            target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.


        Return:
            (1,) or (C,), Calculated dice coefficient, averaged or by labels.
        """
        return differentiable_dice_score(
            input=input,
            target=target,
            bg=self.include_background,
            nan_score=self.nan_score,
            no_fg_score=self.no_fg_score,
            reduction=self.reduction,
            apply_activation=self.apply_activation,
        )
