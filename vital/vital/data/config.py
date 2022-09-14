from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

SemanticStructureId = Union[int, Sequence[int]]


class DataTag(Enum):
    """Extension of Python's ``Enum`` type to provide easy conversion and display methods."""

    def __str__(self):  # noqa: D105
        return self.name.lower()

    def __repr__(self):  # noqa: D105
        return str(self)

    @classmethod
    def names(cls) -> List[str]:
        """Lists the names for all the elements of the enumeration.

        Returns:
            Names of all the elements in the enumeration.
        """
        return [str(e) for e in cls]

    @classmethod
    def values(cls) -> List:
        """Lists the values for all the elements of the enumeration.

        Returns:
            Values of all the elements in the enumeration.
        """
        return [e.value for e in cls]

    @classmethod
    def from_name(cls, name: str) -> "DataTag":
        """Fetches an element of the enumeration based on its name.

        Args:
            name: attribute name of the element in the enumeration.

        Returns:
            Element from the enumeration corresponding to the requested name.
        """
        return cls[name.upper()]


class Subset(DataTag):
    """Enumeration to gather tags referring to commonly used subsets of a whole dataset.

    Attributes:
        TRAIN: Label of the training subset.
        VAL: Label of the validation subset.
        TEST: Label of the testing subset.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Tags:
    """Class to gather the tags referring to the different type of data stored.

    Args:
        id: Tag referring to a unique identifier for the data.
        group: Tag referring to an identifier for the group the data belongs to.
        neighbors: Tag referring to an item's neighbor, provided alongside the item itself.
        img: Tag referring to images.
        gt: Tag referring to groundtruths, used as reference when evaluating models' scores.
        pred: Tag referring to original predictions.
        post_pred: Tag referring to post processed predictions.
        encoding: Tag referring to an encoding of the system's input.
    """

    id: str = "id"
    group: str = "group"
    neighbors: str = "neighbors"
    img: str = "img"
    gt: str = "gt"
    pred: str = "pred"
    post_pred: str = "post_pred"
    encoding: str = "z"


@dataclass(frozen=True)
class DataParameters:
    """Class for defining parameters related to the nature of the data.

    Args:
        in_shape: Shape of the input data (e.g. height, width, channels).
        out_shape: Shape of the target data (e.g. height, width, channels).
        labels: Labels provided with the data, required when using segmentation task APIs.
    """

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    labels: Optional[Tuple[DataTag, ...]] = None
