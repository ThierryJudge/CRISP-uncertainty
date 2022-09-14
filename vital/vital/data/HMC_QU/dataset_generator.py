import argparse
import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import scipy.io
import skvideo
import skvideo.io
from PIL.Image import LINEAR
from skimage import color
from sklearn.model_selection import train_test_split

from vital.data.camus.config import CamusTags, View, img_save_options, seg_save_options
from vital.data.config import Subset
from vital.data.HMC_QU.config import image_size
from vital.utils.image.transform import resize_image

AVAILABLE_GT_SYMBOL = "ü"
AVAILABLE_GT_ROW = "LV Wall Ground-truth Segmentation Masks"
DATASET_INFO_FILENAME = "HMC_QU.xlsx"
IMG_RELATIVE_PATH = Path("HMC-QU Echos/HMC-QU Echos")
GT_RELATIVE_PATH = Path("LV Ground-truth Segmentation Masks")
target_image_size = (256, 256)


def generate_dataset(path: Path, name: Path, seed: int, test_size: float, val_size: float) -> None:
    """Generates the h5 dataset.

    Args:
        path: path to the raw data.
        name: Name of the output file.
        seed: random seed for splitting sets
        test_size: size of the test set with respect to the full dataset
        val_size: size of the validation set with respect to the split train set.
    """
    dataset_info = pd.read_excel(path / DATASET_INFO_FILENAME, index_col="ECHO")
    columns = [
        "SEG1",
        "SEG2",
        "SEG3",
        "SEG5",
        "SEG6",
        "SEG7",
        "Reference Frame",
        "End of Cycle",
        "LV Wall Ground-truth Segmentation Masks",
    ]
    dataset_info.columns = columns
    dataset_info = dataset_info[dataset_info[AVAILABLE_GT_ROW] == AVAILABLE_GT_SYMBOL]  # Only keep samples with GT
    patient_list = list(dataset_info.index.values)

    train_patients, test_patients = train_test_split(patient_list, test_size=test_size, random_state=seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=val_size, random_state=seed)

    with h5py.File(name, "w") as h5f:
        # List all the samples by vendor.

        # Create training sets
        print("Training...")
        print(len(train_patients))
        train_group = h5f.create_group(Subset.TRAIN.value)
        generate_set(train_patients, path, dataset_info, train_group)

        print("Validation...")
        print(len(val_patients))
        val_group = h5f.create_group(Subset.VAL.value)
        generate_set(val_patients, path, dataset_info, val_group)

        print("Testing...")
        print(len(test_patients))
        test_group = h5f.create_group(Subset.TEST.value)
        generate_set(test_patients, path, dataset_info, test_group)


def generate_set(patient_list: List[str], path: Path, dataset_info: pd.DataFrame, group: h5py.Group) -> None:
    """Generates on set.

    Args:
        patient_list: List of patient IDs
        path: path to the raw data.
        dataset_info: dataframe containing dataset information.
        group: group in which to save the set.
    """
    for patient in patient_list:
        img = skvideo.io.vread(str(path / IMG_RELATIVE_PATH / (patient + ".avi")))
        gt = scipy.io.loadmat(str(path / GT_RELATIVE_PATH / ("Mask_" + patient + ".mat")))["predicted"]

        reference_frame = dataset_info.loc[patient]["Reference Frame"]
        end_of_cycle = dataset_info.loc[patient]["End of Cycle"]
        # Subtract one as frames are from Matlab format (index starts at 1)
        img = img[reference_frame - 1 : end_of_cycle]

        img = np.array([color.rgb2gray(resize_image(x, (image_size, image_size), resample=LINEAR)) for x in img])
        gt_proc = np.array([resize_image(y, (image_size, image_size)) for y in gt])

        patient_group = group.create_group(re.split("_|\\s", patient)[0])
        patient_view_group = patient_group.create_group(View.A4C)
        patient_view_group.create_dataset(name=CamusTags.img_proc, data=img, **img_save_options)
        patient_view_group.create_dataset(name=CamusTags.gt_proc, data=gt_proc, **seg_save_options)
        patient_view_group.create_dataset(name=CamusTags.gt, data=gt, **seg_save_options)


def main():
    """Main function where we define the argument for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Script to create the HMC_QU dataset "
            "hdf5 file from the directory. "
            "The given directory need to have two "
            "directory inside, 'training' and 'testing'."
        )
    )
    parser.add_argument(
        "--path", type=Path, required=True, help="Path of the HMC_QU downloaded dataset unziped folder."
    )
    parser.add_argument("--name", type=Path, default=Path("HMC_QU.h5"), help="Name of the generated hdf5 file.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the data split")
    parser.add_argument("--test_size", type=float, default=0.25, help="Size of test set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Size of validation set")

    args = parser.parse_args()

    generate_dataset(args.path, args.name, args.seed, args.test_size, args.val_size)


if __name__ == "__main__":
    main()
