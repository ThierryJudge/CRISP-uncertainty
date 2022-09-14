<div align="center">

# VITAL

Welcome to the repo of the
[Videos & Images Theory and Analytics Laboratory (VITAL)](http://vital.dinf.usherbrooke.ca/ "VITAL home page") of
Sherbrooke University, headed by Professor [Pierre-Marc Jodoin](http://info.usherbrooke.ca/pmjodoin/)

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI: Code Format](https://github.com/nathanpainchaud/vital/actions/workflows/code-format.yml/badge.svg?branch=dev)](https://github.com/nathanpainchaud/vital/actions/workflows/code-format.yml?query=branch%3Adev)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/nathanpainchaud/vital/blob/dev/LICENSE)

</div>

## Description
This repository was not designed to be used as a standalone project, but was rather meant to be used as a third-party
library for more applied projects.

To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- [data](vital/data): utilities to process and interface with common medical image datasets, from processing raw image
files (e.g. `.mhd` or `nii.gz`) to implementations of torchvision's `VisionDataset`.

- [metrics](vital/metrics): common metrics that are not part of the traditional libraries, whether those metrics are
losses for training (see [train](vital/metrics/train)) or scores to evaluate the systems' performance (see
[evaluate](vital/metrics/evaluate)).

- [modules](vital/modules): generic models, organized by task (e.g. [classification](vital/modules/segmentation),
[generative](vital/modules/generative), etc.).

- [systems](vital/systems): common boilerplate Lightning module code (split across mixins), from which concrete
projects' systems should inherit.

- [utils](vital/utils): a wide range of common utilities that may be used in multiple other packages (e.g.
[logging](vital/utils/logging.py), [image processing](vital/utils/image), etc.).

- [VitalRunner](vital/runner.py): common boilerplate code surrounding the use of Lightning's `Trainer` that
handles a generic train and eval run of a model.

## How to use

### Install
To install the project, run the following command from the project's root directory:
```shell script
pip install .
```
> NOTE: This instruction applies when you only want to use the project. If you want to play around with the code and
> contribute to the project, see [the section on how to contribute](#how-to-contribute).

### Hydra

This project relies heavily on [Hydra](https://hydra.cc/). Hydra is a yaml based replacement for the standard Argparser.
The `@hydra.main` decorator is required in the script you want to run.

### Environment variables

### Tracking experiments
By default, Lightning logs runs locally in a format interpretable by
[Tensorboard](https://www.tensorflow.org/tensorboard/).

Another option is to use [Comet](https://www.comet.ml/) to log experiments, either online or offline. To enable the
tracking of experiments using Comet change the instance of the logger in your config or when running your script.
```bash
python <your_runner_script.py> --logger=comet/{online or offline}
```

Note that the online Comet Logger will require the `COMET_API_KEY` environment variable to be set. You must also specify
the `project_name` and `workspace` parameters in your Hydra config.


## How to Contribute

### Environment Setup
If you want to contribute to the project, you must include it differently in your python environment. Once again, it is
recommended to use pip to install the project. However, this time the project should be installed in editable mode, with
the required additional development dependencies:
```shell script
pip install -e .[dev]
```

### Version Control Hooks
Before first trying to commit to the project, it is important to setup the version control hooks, so that commits
respect the coding standards in place for the project. The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file
defines the pre-commit hooks that should be installed in any project contributing to the `vital` repository. To setup
the version control hooks, run the following command:
```shell script
pre-commit install
```

> NOTE: In case you want to copy the pre-commit hooks configuration to your own project, you're welcome to :)
> The configuration file for each hook is located in the following files:
> - [isort](https://github.com/timothycrosley/isort): [`pyproject.toml`](./pyproject.toml), `[tool.isort]` section
> - [black](https://github.com/psf/black): [`pyproject.toml`](./pyproject.toml), `[tool.black]` section
> - [flake8](https://gitlab.com/pycqa/flake8): [`setup.cfg`](./setup.cfg), `[flake8]` section
>
> However, be advised that `isort` must be configured slightly differently in each project. The `src_paths` tag
> should thus reflect the package directory name of the current project, in place of `vital`.
