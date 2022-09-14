import os
from abc import ABC
from typing import Dict, Union

import pytorch_lightning as pl
from torch.utils.data import Dataset

from vital.data.config import DataParameters, Subset


class VitalDataModule(pl.LightningDataModule, ABC):
    """Top-level abstract data module from which to inherit.

    Implementations of behaviors related to data handling (e.g. data preparation) are made through this class.
    """

    def __init__(
            self,
            data_params: DataParameters,
            batch_size: int,
            num_workers: int = os.cpu_count() - 1,
            pin_memory: bool = True,
            **kwargs):
        """Initializes class instance.

        References:
            - ``workers`` documentation, for more detail:
              https://pytorch-lightning.readthedocs.io/en/stable/benchmarking/performance.html#num-workers

        Args:
            data_params: Parameters related to the data necessary to initialize networks working with this dataset.
            batch_size: Size of batches.
            num_workers: Number of subprocesses to use for data loading.
                ``workers=0`` means that the data will be loaded in the main process.
        """
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._dataset: Dict[Subset, Dataset] = {}

    def dataset(self, subset: Subset = None) -> Union[Dict[Subset, Dataset], Dataset]:
        """Returns the subsets of the data (e.g. train) and their torch ``Dataset`` handle.

        It should not be called before ``setup``, when the datasets are set.

        Args:
            subset: Specific subset for which to get the ``Dataset`` handle.

        Returns:
            If ``subset`` is provided, returns the handle to a specific dataset. Otherwise, returns the mapping between
            subsets of the data (e.g. train) and their torch ``Dataset`` handle.
        """
        if subset is not None:
            return self._dataset[subset]

        return self._dataset
