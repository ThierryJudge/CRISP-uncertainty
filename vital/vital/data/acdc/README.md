# Description

This folder contains the necessary file to use the ACDC dataset: [ACDC](www.creatis.insa-lyon
.fr/Challenge/acdc/) 2017 dataset (Automatic Cardiac Delineation Challenge).

# How to run

## Dataset Generator

The script can be used to generate the semantic segmentation dataset into the hdf5 format, from the nifti MRIs and
groundtruths.

The raw data must be downloaded and organized in the following format

```yaml
/:  # root

    - training :
        - patient001:
            - Info.cfg
            - patient002_frame{ED}.nii.gz
            - patient002_frame{ED}_gt.nii.gz
            - patient002_frame{ES}.nii.gz
            - patient002_frame{ES}_gt.nii.gz
        ...
        - patient100: ...

    - testing:  # patient data
        - patient101: ...
        ...
        - patient150: ...

```
Once you have downloaded the dataset and extracted its content to the format mentioned above:

```bash
python dataset_generator.py --path='path/of/the/raw_MRI_nifti_data' --name='name/of/the/output.hdf5'
```


### Options
To list all the options available when generating a dataset, run:
```bash
python dataset_generator.py -h
```

### Format
Once you've finished generating the dataset, it can be used through the
[ACDC `VisionDataset` interface](dataset.py) to train and evaluate models. The data inside in the HDF5 dataset is
structured according to the following format:
```yaml
/:  # root of the file
    - prior: Average shape over all training patients
    - train:
        - patient{XXXX}:
          - ED:
              - img # MRI image (N, 256, 256, 1)
              - gt # segmentation (N x 256 x 256)
          - ES:
              - img # MRI image
              - gt # segmentation
        ....
        - patient{YYYY}: ...
    - val:
        - patient{XXXX}:
        ....
        - patient{YYYY}: ...
    - test:
        - patient{XXXX}:
        ....
        - patient{YYYY}: ...

```
