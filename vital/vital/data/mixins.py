from abc import abstractmethod
from typing import List

from vital.data.config import Subset
from vital.data.data_module import VitalDataModule


class StructuredDataMixin(VitalDataModule):
    """``VitalDataModule`` mixin for datasets where data has more structure than only the target labels."""

    @abstractmethod
    def group_ids(self, subset: Subset, *args, **kwargs) -> List[str]:
        """Lists the IDs of the different groups/clusters samples in the data can belong to.

        Args:
            *args: Positional arguments that parameterize the requested data structure.
            **kwargs: Keyword arguments that parameterize the requested data structure.

        Returns:
            IDs of the different groups/clusters samples in the data can belong to.
        """
