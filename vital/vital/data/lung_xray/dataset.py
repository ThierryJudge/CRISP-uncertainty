import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union, Optional

import albumentations as A
import cv2
import h5py
import numpy as np
import torchvision
from torch import Tensor
from torchvision.datasets import VisionDataset
from vital.data.camus.data_struct import PatientData, ViewData
from vital.data.config import Subset, Tags
from vital.data.lung_xray.config import View
from vital.data.transforms import DiffusedNoise
from vital.utils.decorators import squeeze
from vital.utils.image.transform import segmentation_to_tensor


class LungXRay(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the Shenzen Montgomery dataset."""

    def __init__(
            self,
            path: Path,
            image_set: Subset,
            predict: bool = False,
            transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
            transform: Callable[[Tensor], Tensor] = None,
            target_transform: Callable[[Tensor], Tensor] = None,
            max_patients: Optional[int] = None,
            data_augmentation: bool = False,
            test_all: bool = False
    ):
        """Initializes class instance.

        Args:
            path: Path to the HDF5 dataset.
            fold: ID of the cross-validation fold to use.
            image_set: Subset of images to use.
            labels: Labels of the segmentation classes to take into account.
            use_sequence: Whether to use the complete sequence between ED and ES for each view.
            predict: Whether to receive the data in a format fit for inference (``True``) or training (``False``).
            neighbors: Neighboring frames to include in a train/val item. The value either indicates the number of
                neighboring frames on each side of the item's frame (`int`), or a list of offsets w.r.t the item's
                frame (`Sequence[int]`).
            neighbor_padding: Mode used to determine how to pad neighboring instants at the beginning/end of a sequence.
                The options mirror those of the ``mode`` parameter of ``numpy.pad``.
            transforms: Function that takes in an input/target pair and transforms them in a corresponding way.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
            transform: Function that takes in an input and transforms it.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
            target_transform: Function that takes in a target and transforms it.
                (only applied when `predict` is `False`, i.e. in train/validation mode)

        Raises:
            RuntimeError: If flags/arguments are requested that cannot be provided by the HDF5 dataset.
                - ``use_sequence`` flag is active, while the HDF5 dataset doesn't include full sequences.
        """
        super().__init__(path, transforms=transforms, transform=transform, target_transform=target_transform)
        self.image_set = image_set
        self.predict = predict
        self.max_patients = max_patients
        self.test_all = test_all
        if transforms is not None:
            self.transforms = A.Compose(transforms)
        elif data_augmentation:
            self.transforms = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                ]
            )

        else:
            self.transforms = None

        self._base_transform = torchvision.transforms.ToTensor()
        self._base_target_transform = segmentation_to_tensor

        # Determine whether to return data in a format suitable for training or inference
        self.item_list = self.list_items()
        self.getter = self._get_test_item if self.predict else self._get_train_item

    def __getitem__(self, index) -> Union[Dict, PatientData]:
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

    def list_items(self) -> List[str]:
        """Lists the paths of the different items.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        image_sets = [self.image_set.value] if not self.test_all else [set.value for set in Subset]
        with h5py.File(self.root, "r") as dataset:
            # List the patients
            items = []
            for set in image_sets:
                items.extend([f"{set}/{x}" for x in dataset[set].keys()])

            items = items[:self.max_patients] if self.max_patients is not None else items

        return items

    def _get_train_item(self, index: int) -> Dict:
        """Fetches data and metadata related to an instant (single image/groundtruth pair + metadata).

        Args:
            index: Index of the instant sample in the train/val set's ``self.item_list``.

        Returns:
            Data and metadata related to an instant.
        """
        item_key = self.item_list[index]

        # Collect data
        with h5py.File(self.root, "r") as dataset:
            img, gt = self._get_data(dataset, item_key, Tags.img, Tags.gt)

        img = img / 255
        gt[gt != 0] = 1

        if self.transforms:
            transformed = self.transforms(image=img, mask=gt)
            img = transformed["image"]
            gt = transformed["mask"]

        img, gt = self._base_transform(img), self._base_target_transform(gt)

        return {
            Tags.id: f"{item_key}",
            Tags.img: img,
            Tags.gt: gt,
        }

    def _get_test_item(self, index: int) -> PatientData:
        """Fetches data required for inference on a test item, i.e. a patient.

        Args:
            index: Index of the test sample in the test set's ``self.item_list``.

        Returns:
            Data related a to a test item, i.e. a patient.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_data = PatientData(id=os.path.basename(self.item_list[index]))
            item_key = self.item_list[index]

            # Collect and process data
            img, gt = LungXRay._get_data(dataset, item_key, Tags.img, Tags.gt)

            img = img / 255
            gt[gt != 0] = 1

            if self.transforms:
                transformed = self.transforms(image=img, mask=gt)
                img = transformed["image"]
                gt = transformed["mask"]

            img_tensor = self._base_transform(img)
            gt_tensor = self._base_target_transform(gt)

            patient_data.views[View.PA] = ViewData(
                img_proc=img_tensor[None],
                gt=gt[None],
                gt_proc=gt_tensor[None],
                voxelspacing=(1, 1, 1),
                instants={'': 0}
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

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--predict", action="store_true")
    params = args.parse_args()

    ds = LungXRay(Path(params.path), image_set=Subset.TEST, predict=params.predict, test_all=True)

    samples = []
    for sample in ds:
        samples.append(sample[Tags.gt].squeeze().numpy())

    samples = np.array(samples)

    print(samples.shape)

    print(samples.min())
    print(samples.max())
    print(samples.mean())
    print(samples.std())

    mean = np.mean(samples, axis=0)

    plt.imshow(mean)
    plt.savefig('jstr.png')
    plt.show()

    exit(0)

    if params.predict:
        patient = ds[random.randint(0, len(ds) - 1)]
        instant = patient.views[View.PA]
        img = instant.img_proc
        gt = instant.gt_proc
        print("Image shape: {}".format(img.shape))
        print("GT shape: {}".format(gt.shape))
        print("ID: {}".format(patient.id))

        img = img.squeeze()
    else:
        sample = ds[0]  # random.randint(0, len(ds) - 1)]
        img = sample[Tags.img].squeeze()
        gt = sample[Tags.gt]
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
