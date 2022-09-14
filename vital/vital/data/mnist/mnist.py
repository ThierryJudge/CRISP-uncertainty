from typing import Any, Dict

import torchvision

from vital.data.config import Tags


class MNIST(torchvision.datasets.MNIST):
    """Wrapper for the MNIST dataset to return dict instead of tuple for each sample."""

    def __getitem__(self, index: int) -> Dict[str, Any]:  # noqa: D105
        img, target = super().__getitem__(index)
        d = {Tags.img: img, Tags.gt: target}
        return d
