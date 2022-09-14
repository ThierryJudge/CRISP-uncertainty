import argparse
import logging
import os
from glob import glob
from os.path import basename
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
from natsort import natsorted
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm

from vital.data.acdc.config import AcdcTags, Instant, Label, image_size
from vital.data.acdc.utils.register import AcdcRegisteringTransformer
from vital.data.config import Subset
from vital.utils.format.numpy import to_categorical, to_onehot
from vital.utils.image.measure import Measure
from vital.utils.image.transform import centered_resize
from vital.utils.logging import configure_logging

logger = logging.getLogger(__name__)


PRIOR_SIZE = 100
PRIOR_HALF_SIZE = PRIOR_SIZE // 2

ROTATIONS = [-60, -45, -15, 0, 15, 45, 60]


def extract_image_paths(root_dir: Path) -> List[Path]:
    """Generate the list of NIFTI images from the data's root directory.

    Args:
        root_dir: Path of the folder containing the ACDC dataset's raw data.

    Returns:
        List of paths to NIFTI images
    """
    if not root_dir.exists():
        return []
    glob_exp = root_dir / "*" / "*"
    paths = natsorted(glob(str(glob_exp)))
    paths = [Path(path) for path in paths if "Info" not in path and path.find("_4d") < 0]
    return paths


def _generate_centered_prob_map(image: np.ndarray, shape: np.ndarray, center: np.ndarray, label: int) -> np.array:
    """Extracts the information from the ground truth image given the centers.

    Args:
        image: (N, H, W) Stack of 2D segmentation of MRI axial slices.
        shape: Height and width of the desired prior image.
        center: (N, 2) Centers of mass for each slice of the probability map.
        label: Label to extract from the segmentation.

    Returns:
        (N, shape, 1) Probability map based on the ground truth image and centers of mass.
    """
    image = np.equal(image, label)[..., None]
    res = np.zeros(shape)
    # Nearest neighbour slice index between the number of slice
    # of the image and the ground truth
    space = np.linspace(0, shape[0] - 1, num=image.shape[0]).astype(np.int32)
    for i, (s, c) in enumerate(zip(space, center)):
        res[s] += image[
            i,
            c[0] - PRIOR_HALF_SIZE : c[0] + PRIOR_HALF_SIZE,
            c[1] - PRIOR_HALF_SIZE : c[1] + PRIOR_HALF_SIZE,
        ]
    return res


def generate_probability_map(dataset: h5py.File, group: h5py.Group, data_augmentation: bool):
    """Generates the probability map from all non-rotated training examples.

    Args:
        dataset: Handle of the hdf5 file containing all the dataset.
        group: Group where to create the prior shape. One of either train, valid or test.
            If in doubt, train should be the default.
        data_augmentation: Indicates if data augmentation was used.
    """
    patient_keys = [key for key in group.keys() if key.endswith("_0")] if data_augmentation else list(group.keys())
    image_keys = []
    for k1 in patient_keys:
        for k2 in group[k1].keys():
            image_keys.append(f"{k1}/{k2}/{AcdcTags.gt}")

    images = [group[k][:] for k in image_keys]

    images_center = [Measure.structure_center(img, 3).astype(np.int16) for img in images]

    prior_shape = np.array([15, PRIOR_SIZE, PRIOR_SIZE, 1])

    prob_maps = []
    for label in Label:
        prob_map = np.array(
            [
                _generate_centered_prob_map(np.copy(img), prior_shape, center, label.value)
                for img, center in tqdm(zip(images, images_center), desc=str(label), total=len(images))
            ]
        )
        prob_maps.append(prob_map.sum(axis=0))

    p_img = np.concatenate(tuple(prob_maps), axis=-1)
    p_img /= p_img.sum(axis=-1, keepdims=True).astype(np.float32)
    dataset.create_dataset("prior", data=p_img[:, :, :, 1:])


def load_instant_data(img_path: Path, gt_path: Optional[Path] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple]:
    """Loads data, gt (when available) and voxel_size spacing from NIFTI file.

    Args:
        img_path: path to image NIFTI file.
        gt_path: path to gt NIFTI file.

    Returns:
        img (N, H, W, 1), gt (N, H, W) and voxel_size (3,) spacing for one instant.
    """
    ni_img = nib.load(img_path)

    img = ni_img.get_fdata().astype(np.float32)
    img = img.transpose(2, 0, 1)[..., np.newaxis]

    img = centered_resize(img, (image_size, image_size))

    voxel_size = ni_img.header.get_zooms()

    if gt_path:
        ni_img = nib.load(gt_path)
        gt = ni_img.get_fdata()
        gt = gt.transpose(2, 0, 1)[..., np.newaxis]

        gt = to_onehot(gt, len(Label))

        gt = centered_resize(gt, (image_size, image_size))

        # Need to redo the background class due to resize
        # that set the image border to 0
        summed = np.clip(gt[..., 1:].sum(axis=-1), 0, 1)
        gt[..., 0] = np.abs(1 - summed)

        # Put data to categorical format.
        gt = to_categorical(gt)

        return img, gt, voxel_size

    return img, None, voxel_size


def write_instant_group(
    instant_group: h5py.Group,
    img_data: np.ndarray,
    gt_data: np.ndarray,
    voxel_size: Tuple,
    rotation: float,
    registering_transformer: Optional[AcdcRegisteringTransformer],
):
    """Writes the data related to an instant to its HDF5 group.

    Args:
        instant_group: HDF5 group in which to save the instant's data
        img_data: Array (N, H, W, 1) containing the img data
        gt_data: Array (N, H, W) containing the gt data
        voxel_size: Voxel size of the instant (3,)
        rotation: Rotation for data augmentation
        registering_transformer: Transformer used for registration
    """
    instant_group.attrs["voxel_size"] = voxel_size

    r_img = rotate(img_data, rotation, axes=(1, 2), reshape=False)
    r_img = np.clip(r_img, img_data.min(), img_data.max())
    r_img[np.isclose(r_img, 0.0)] = 0.0

    if registering_transformer is not None:
        registering_parameters, gt_data, r_img = registering_transformer.register_batch(gt_data, r_img)
        instant_group.attrs.update(registering_parameters)

    instant_group.create_dataset(AcdcTags.img, data=r_img)

    if gt_data is not None:
        r_img = rotate(gt_data, rotation, axes=(1, 2), output=np.uint8, reshape=False)
        instant_group.create_dataset(AcdcTags.gt, data=r_img)


def create_database_structure(
    group: h5py.Group,
    data_augmentation: bool,
    registering: bool,
    data_ed: Path,
    gt_ed: Path,
    data_es: Path,
    gt_es: Path,
    data_mid: Optional[Path] = None,
    gt_mid: Optional[Path] = None,
):
    """Creates the dataset for the End-Systolic and End-Diastolic phases.

    If some data augmentation is involved we create also the rotation for each phase.

    Args:
        group: Group where we add each image by its name and the rotation associated to it.
        data_augmentation: Enable/Disable data augmentation.
        registering: Enable/Disable registering.
        data_ed: Path of the NIFTI diastolic MRI image.
        gt_ed: Path of the NIFTI diastolic MRI segmentation of `data_ed`.
        data_es: Path of the NIFTI systolic MRI image.
        gt_es: Path of the NIFTI systolic MRI segmentation of `data_es`.
        data_mid: Path of the NIFTI MRI image at a time step between ED and ES.
        gt_mid: Path of the NIFTI MRI segmentation of `data_mid`.

    """
    p_name = data_ed.parent.name

    ed_img, edg_img, ed_voxel_size = load_instant_data(data_ed, gt_ed)
    es_img, esg_img, es_voxel_size = load_instant_data(data_es, gt_es)

    if data_mid:
        mid_img, midg_img, mid_voxel_size = load_instant_data(data_mid, gt_mid)

    if data_augmentation:
        rotations = ROTATIONS
    else:
        rotations = [
            0,
        ]

    if registering:
        registering_transformer = AcdcRegisteringTransformer()
    else:
        registering_transformer = None

    for rotation in rotations:
        name = f"{p_name}_{rotation}" if len(rotations) > 1 else p_name
        patient = group.create_group(name)

        write_instant_group(
            patient.create_group(Instant.ED.value), ed_img, edg_img, ed_voxel_size, rotation, registering_transformer
        )
        write_instant_group(
            patient.create_group(Instant.ES.value), es_img, esg_img, es_voxel_size, rotation, registering_transformer
        )

        # Add mid-cycle data
        if data_mid:
            write_instant_group(
                patient.create_group(Instant.MID.value),
                mid_img,
                midg_img,
                mid_voxel_size,
                rotation,
                registering_transformer,
            )


def generate_dataset(data_path: Path, output_path: Path, data_augmentation: bool = False, registering: bool = False):
    """Generates each dataset, train, valid and test.

    Args:
        data_path: Root directory of the raw data downloaded from the ACDC challenge
        output_path: Path of the HDF5 file to output.
        data_augmentation: Enable/Disable data augmentation.
        registering: Enable/Disable registering.

    Raises:
        ValueError: If provided file names don't match.
    """
    if data_augmentation:
        logger.info("Data augmentation enabled, rotation " "from -60 to 60 by step of 15.")
    if registering:
        logger.info("Registering enabled, MRIs and groundtruths centered and rotated.")
    rng = np.random.RandomState(1337)

    # get training examples
    train_paths = extract_image_paths(data_path / "training")
    # We have 4 path, path_ED, path_gt_ED, path_ES and path_gt_ES
    train_paths = np.array(list(zip(train_paths[0::4], train_paths[1::4], train_paths[2::4], train_paths[3::4])))

    # 20 is the number of patients per group
    patients_per_group = 20
    indexes = np.arange(patients_per_group)

    train_idxs = []
    valid_idxs = []
    # 5 is the number of groups
    for i in range(5):
        start = i * patients_per_group
        idxs = indexes + start
        rng.shuffle(idxs)
        t_idxs = idxs[: int(indexes.shape[0] * 0.75)]
        v_idxs = idxs[int(indexes.shape[0] * 0.75) :]
        train_idxs.append(t_idxs)
        valid_idxs.append(v_idxs)

    train_idxs = np.array(train_idxs).flatten()
    valid_idxs = np.array(valid_idxs).flatten()
    valid_paths = train_paths[valid_idxs]
    train_paths = train_paths[train_idxs]

    # get testing examples
    if os.path.exists(os.path.join(data_path, "testing_with_gt_mid")):
        test_paths = extract_image_paths(data_path / "testing_with_gt_mid")
        test_paths = np.array(
            list(
                zip(
                    test_paths[0::6],
                    test_paths[1::6],
                    test_paths[2::6],
                    test_paths[3::6],
                    test_paths[4::6],
                    test_paths[5::6],
                )
            )
        )
    else:
        test_paths = extract_image_paths(data_path / "testing_with_gt")
        test_paths = np.array(list(zip(test_paths[0::4], test_paths[1::4], test_paths[2::4], test_paths[3::4])))
        test_paths = np.array([np.insert(i, 2, [None, None]).tolist() for i in test_paths])

    with h5py.File(output_path, "w") as output_dataset:

        output_dataset.attrs[AcdcTags.registered] = registering

        # Training samples ###
        group = output_dataset.create_group(Subset.TRAIN.value)
        for p_ed, g_ed, p_es, g_es in tqdm(train_paths, desc="Training"):
            # Find missmatch in the zip
            if str(p_ed) != str(g_ed).replace("_gt", "") or str(p_es) != str(g_es).replace("_gt", ""):
                raise ValueError(f"File name don't match: {p_ed} instead of {g_ed}, {p_es} instead of {g_es}")

            create_database_structure(group, data_augmentation, registering, p_ed, g_ed, p_es, g_es)

        # Generate the probability map from the ground truth training examples
        generate_probability_map(output_dataset, group, data_augmentation)

        # Validation samples ###
        group = output_dataset.create_group(Subset.VAL.value)
        for p_ed, g_ed, p_es, g_es in tqdm(valid_paths, desc="Validation"):
            # Find missmatch in the zip
            if str(p_ed) != str(g_ed).replace("_gt", "") or str(p_es) != str(g_es).replace("_gt", ""):
                raise ValueError(f"File name don't match: {p_ed} instead of {p_ed}, {p_es} instead of {g_es}.")

            create_database_structure(group, False, registering, p_ed, g_ed, p_es, g_es)

        # Testing samples ###
        group = output_dataset.create_group(Subset.TEST.value)
        for p_ed, g_ed, p_mid, g_mid, p_es, g_es in tqdm(test_paths, desc="Testing"):
            p_mid = None if p_mid == "None" else p_mid
            g_mid = None if g_mid == "None" else g_mid
            # Find missmatch in the zip
            if str(basename(p_ed)) != str(basename(g_ed)).replace("_gt", "") or str(basename(p_es)) != str(
                basename(g_es)
            ).replace("_gt", ""):
                raise ValueError(f"File name don't match: {p_ed} instead of {g_ed}, {p_es} instead of {g_es}.")
            create_database_structure(
                group, data_augmentation, registering, p_ed, g_ed, p_es, g_es, data_mid=p_mid, gt_mid=g_mid
            )


def main():
    """Main function where we define the argument for the script."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Script to create the ACDC dataset hdf5 file from the directory.The given directory need to have two "
            "directory inside, 'training' and 'testing'. "
        )
    )
    parser.add_argument("data_path", type=Path, help="Path to the root directory of the downloaded ACDC data")
    parser.add_argument("--output_path", type=Path, default="acdc.h5", help="Path to the HDF5 file to generate")
    data_processing_group = parser.add_mutually_exclusive_group()
    data_processing_group.add_argument(
        "-d", "--data_augmentation", action="store_true", help="Add data augmentation (rotation -60 to 60)."
    )
    data_processing_group.add_argument(
        "-r",
        "--registering",
        action="store_true",
        help="Apply registering (registering and rotation)." "Only works when groundtruths are provided.",
    )
    args = parser.parse_args()
    generate_dataset(args.data_path, args.output_path, args.data_augmentation, args.registering)


if __name__ == "__main__":
    main()
