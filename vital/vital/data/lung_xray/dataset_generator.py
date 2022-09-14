import os
import glob
from matplotlib import pyplot as plt
import imageio
from os.path import join as pjoin

from pytorch_lightning import seed_everything
from skimage import color
from skimage.transform import resize
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from vital.data.config import Tags, Subset
from vital.data.lung_xray.config import img_save_options, seg_save_options


def write_group_hdf5(group, file_list, img_folder_path, gt_folder_path, img_size):
    for _, file in tqdm(enumerate(file_list), total=len(file_list)):
        file = os.path.basename(file)

        file_name = os.path.splitext(file)[0]
        img_path = pjoin(img_folder_path, file)
        gt_path = pjoin(gt_folder_path, "{}_mask.png".format(file_name))

        if os.path.isfile(gt_path):
            sample_group = group.create_group(file_name)

            img = imageio.imread(img_path)
            gt = imageio.imread(gt_path)

            if img.ndim == 3:
                img = color.rgb2gray(img)

            img_resize = resize(img, (img_size, img_size), preserve_range=True)
            gt_resize = resize(gt, (img_size, img_size), preserve_range=True)

            sample_group.create_dataset(name=Tags.img, data=img_resize, **img_save_options)
            sample_group.create_dataset(name=Tags.gt, data=gt_resize, **seg_save_options)


def write_hdf5(raw_data_path, h5_name, test_split, val_split, img_size, split_source):
    xray_folder_path = pjoin(raw_data_path, 'CXR_png')
    gt_folder_path = pjoin(raw_data_path, 'masks')

    files = glob.glob(pjoin(xray_folder_path, '*.png'))

    if split_source:
        china_files = [f for f in files if "CHN" in f]
        montgomery_files = [f for f in files if "MCU" in f]
        train_files, test_files = china_files, montgomery_files
    else:

        train_files, test_files = train_test_split(files, test_size=test_split)
    train_files, val_files = train_test_split(files, test_size=val_split)

    print(f"Number of train files {len(train_files)}")
    print(f"Number of validation files {len(val_files)}")
    print(f"Number of test files {len(test_files)}")

    f = h5py.File(h5_name, 'w')

    write_group_hdf5(f.create_group(Subset.TRAIN.value), train_files, xray_folder_path, gt_folder_path, img_size)
    write_group_hdf5(f.create_group(Subset.VAL.value), val_files, xray_folder_path, gt_folder_path, img_size)
    write_group_hdf5(f.create_group(Subset.TEST.value), test_files, xray_folder_path, gt_folder_path, img_size)


"""
Data must be in one folder 
    -data
        -masks
        -CXR_png
Data can be downloaded from: https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Root directory under which the raw data is stored")
    parser.add_argument("--h5_name", type=str, default="lung_xray.h5",
                        help="Name of the hdf5 file in which to save the data")
    parser.add_argument("--test_split", type=float, help="Test set split", default=0.15)
    parser.add_argument("--val_split", type=float, help="Validation set split", default=0.1)
    parser.add_argument("--split_source", action='store_true', help="Split train test split with image origin")
    parser.add_argument("--img_size", type=int, help="Size of resized images", default=256)
    parser.add_argument("--seed", type=int, help="Seed", default=0)

    args = parser.parse_args()

    seed_everything(args.seed)

    write_hdf5(raw_data_path=args.data_path, h5_name=args.h5_name, test_split=args.test_split,
               val_split=args.val_split, img_size=args.img_size, split_source=args.split_source)
