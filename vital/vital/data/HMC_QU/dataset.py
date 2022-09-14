import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import to_tensor

from vital.data.camus.config import CamusTags
from vital.data.camus.data_struct import PatientData, ViewData
from vital.data.config import Subset
from vital.data.transforms import NormalizeSample
from vital.utils.decorators import squeeze
from vital.utils.image.transform import segmentation_to_tensor

ItemId = Tuple[str, int]


class HMC_QU(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the HMC_QU dataset."""

    def __init__(
        self,
        path: Path,
        image_set: Subset,
        predict: bool = False,
        transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
        transform: Callable[[Tensor], Tensor] = None,
        target_transform: Callable[[Tensor], Tensor] = None,
    ):
        """Initializes class instance.

        Args:
            path: Path to the HDF5 dataset.
            image_set: Subset of images to use.
            predict: Whether to receive the data in a format fit for inference (``True``) or training (``False``).
            transforms: Function that takes in an input/target pair and transforms them in a corresponding way.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
            transform: Function that takes in an input and transforms it.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
            target_transform: Function that takes in a target and transforms it.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
        """
        super().__init__(path, transforms=transforms, transform=transform, target_transform=target_transform)
        self.image_set = image_set

        self.predict = predict

        print(transform)

        # Determine whether to return data in a format suitable for training or inference
        if self.predict:
            self.item_list = self.list_groups(level="patient")
            self.getter = self._get_test_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index) -> Union[Dict[str, Union[str, Tensor]], PatientData]:
        """Fetches an item, whose structure depends on the ``predict`` value, from the internal list of items.

        Notes:
            - When in ``predict`` mode (i.e. for test-time inference), an item corresponds to the views' ultrasound
              images and groundtruth segmentations for a patient.
            - When not in ``predict`` mode (i.e. during training), an item corresponds to an image/segmentation pair for
              a single frame.

        Args:
            index: Index of the item to fetch from the internal sequence of items.

        Returns:
            Item from the internal list at position ``index``.
        """
        return self.getter(index)

    def __len__(self):  # noqa: D105
        return len(self.item_list)

    def list_groups(self, level: Literal["patient", "view"] = "view") -> List[str]:
        """Lists the paths of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'view': all the data from the same view of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        with h5py.File(self.root, "r") as dataset:
            # List the patients
            groups = list(dataset[self.image_set.value].keys())
            if level == "view":
                groups = [f"{patient}/{view}"
                          for patient in groups
                          for view in dataset[f"{self.image_set}/{patient}"].keys()
                          ]

        return groups

    def _get_instant_paths(self) -> List[ItemId]:
        """Lists paths to the instants, from the requested ``self.image_set``, inside the HDF5 file.

        Returns:
            Paths to the instants, from the requested ``self.image_set``, inside the HDF5 file.
        """
        image_paths = []
        view_paths = self.list_groups(level="view")
        with h5py.File(self.root, "r") as dataset:
            for view_path in view_paths:
                view_group = dataset[f"{self.image_set}/{view_path}"]
                for instant in range(view_group[CamusTags.gt].shape[0]):
                    image_paths.append((view_path, instant))
        return image_paths

    def _get_train_item(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Fetches data required for training on a train/val item (single image/groundtruth pair).

        Args:
            index: Index of the train/val sample in the train/val set's ``self.item_list``.

        Returns:
            Data for training on a train/val item.
        """
        patient_view_key, instant = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect and process data
            view_imgs, view_gts = self._get_data(dataset, f"{self.image_set}/{patient_view_key}",
                                                 CamusTags.img_proc, CamusTags.gt_proc)
            img = view_imgs[instant]
            gt = view_gts[instant]

            # Collect metadata
            # Explicit cast to float32 to avoid "Expected object" type error in PyTorch models
            # that output ``FloatTensor`` by default (and not ``DoubleTensor``)
            frame_pos = np.float32(instant / view_imgs.shape[0])

        img, gt = to_tensor(img), segmentation_to_tensor(gt)
        if self.transforms:
            img, gt = self.transforms(img, gt)
        frame_pos = torch.tensor([frame_pos])

        return {
            CamusTags.id: f"{patient_view_key}/{instant}",
            CamusTags.group: patient_view_key,
            CamusTags.img: img,
            CamusTags.gt: gt,
            CamusTags.frame_pos: frame_pos,
            CamusTags.voxelspacing: np.array((1, 1))
        }

    def _get_test_item(self, index: int) -> PatientData:
        """Fetches data required for inference on a test item, i.e. a patient.

        Args:
            index: Index of the test sample in the test set's ``self.item_list``.

        Returns:
            Data related a to a test item, i.e. a patient.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_data = PatientData(id=self.item_list[index])
            for view in dataset[f"{self.image_set}/{self.item_list[index]}"]:
                patient_view_key = f"{self.image_set}/{self.item_list[index]}/{view}"

                # Collect and process data
                proc_imgs, proc_gts, gts = self._get_data(
                    dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc, CamusTags.gt
                )

                # Transform arrays to tensor
                proc_imgs_tensor = torch.stack([to_tensor(proc_img) for proc_img in proc_imgs])
                proc_gts_tensor = torch.stack([segmentation_to_tensor(proc_gt) for proc_gt in proc_gts])

                # Extract metadata concerning the registering applied

                # Compute attributes for the sequence
                attrs = {CamusTags.frame_pos: torch.linspace(0, 1, steps=len(proc_imgs)).unsqueeze(1)}

                patient_data.views[view] = ViewData(
                    img_proc=proc_imgs_tensor,
                    gt_proc=proc_gts_tensor,
                    gt=gts,
                    voxelspacing=(1.0, 1.0, 1.0),
                    instants={str(i): i for i in range(proc_imgs_tensor.shape[0])},
                    attrs=attrs,
                    registering=None,
                )

        return patient_data

    @staticmethod
    @squeeze
    def _get_data(file: h5py.File, patient_view_key: str, *data_tags: str) -> List[np.ndarray]:
        """Fetches the requested data for a specific patient/view dataset from the HDF5 file.

        Args:
            file: HDF5 dataset file.
            patient_view_key: `patient/view` access path of the desired view group.
            *data_tags: Names of the datasets to fetch from the view.

        Returns:
            Dataset content for each tag passed in the parameters.
        """
        patient_view = file[patient_view_key]
        return [patient_view[data_tag][()] for data_tag in data_tags]


if __name__ == "__main__":
    import random
    from argparse import ArgumentParser

    from matplotlib import pyplot as plt

    from vital.data.camus.config import View

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--predict", action="store_true")
    params = args.parse_args()

    ds = HMC_QU(Path(params.path), image_set=Subset.TRAIN, predict=params.predict) #, transform=Normalize(mean=0.1555, std=0.2126))

    samples = []
    for sample in ds:
        samples.append(sample[CamusTags.img].squeeze().numpy())

    samples = np.array(samples)

    print(samples.min())
    print(samples.max())
    print(samples.mean())
    print(samples.std())

    if params.predict:
        patient = ds[random.randint(0, len(ds) - 1)]
        instant = patient.views[View.A4C]
        img = instant.img_proc
        gt = instant.gt_proc
        print("Image shape: {}".format(img.shape))
        print("GT shape: {}".format(gt.shape))
        print("ID: {}".format(patient.id))

        slice = random.randint(0, len(img) - 1)
        img = img[slice].squeeze()
        gt = gt[slice]
    else:
        sample = ds[10]
        img = sample[CamusTags.img].squeeze()
        print(img.min())
        print(img.max())
        gt = sample[CamusTags.gt]
        print("Image shape: {}".format(img.shape))
        print("GT shape: {}".format(gt.shape))


    print(img.min())
    print(img.max())
    print(img.mean())
    print(img.std())

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(gt)
    plt.show(block=False)

    plt.figure(2)
    plt.imshow(img, cmap="gray")
    plt.imshow(gt, alpha=0.2)
    plt.show()
