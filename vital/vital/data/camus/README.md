## Description

CAMUS is short for Cardiac Acquisitions for Multi-structure Ultrasound Segmentation
(also a famous French writer / philosopher).

- Ultrasound 2D images of the heart at end diastole (ED) and end systole (ES) on 2CH and 4CH views.
It contains as of yet sequences from 500 patients. It may be completed with additional data with the intention of
organizing an international challenge.

- This database is the result of a one-year collaboration between Olivier Bernard, Sarah Leclerc, and Florian Espinoza.
It has been involved in a collaborative study between the universities of Lyon (France), NTNU (Norway),
Sherbrooke (Canada) and Leuven (Belgium).

- The data can be efficiently split in 10 balanced subfolds with respect to the ejection fraction and the image quality
(see txt files).

The dataset itself is available from the [challenge web page](https://www.creatis.insa-lyon.fr/Challenge/camus/).

## How to run

### Cross validation
Once you have downloaded the dataset and extracted its content from the archive:
```bash
# list cross validation HDF5 dataset generation options
python dataset_generator.py -h

# generate the cross validation HDF5 dataset with default options
# NOTE: this command could take a bit of time to finish
#       (up to 2 hours depending on the chosen options)
python dataset_generator.py {path to the extracted data}
```
Once you've finished generating the cross validation dataset, it can be used through the
[CAMUS `VisionDataset` interface](dataset.py) to train and evaluate models. The data inside in the HDF5 dataset is
structured according to the following format:
```yaml
/:  # root of the file

    - cross_validation:  # dedicated group for the folds' metadata
        - fold_{n}:
            - train  # list of patients in the training set of this fold
            - val  # list of patients in the validation set of this fold
            - test  # list of patients in the testing set of this fold
        - fold_{m}: ...
        ...

    - patient{XXXX}:  # patient data
        - {2|4}CH:  # sequence data w/ some metadata attributes
                    # (e.g. voxel size, ED/ES frames, etc.)
            - img_proc  # NxHxW, raw image after pre-processing
                        # (e.g. resized to a fixed size)
            - gt_proc  # NxHxW, reference segmentation after pre-processing
                       # (e.g. resized to a fixed size)
            - gt  # NxOG_HxOG_W, copy of the original reference segmentation
        - {2|4}CH: ...
    - patient{YYYY}: ...
    ...
```
