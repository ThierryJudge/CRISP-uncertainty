import os
import random
from typing import Tuple

import comet_ml  # noqa
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from crisp_uncertainty.system.uncertainty import UncertaintyMapEvaluationSystem
from torch import Tensor
from torch.nn import functional as F
from vital.data.config import DataParameters, Subset
from vital.systems.segmentation import SegmentationComputationMixin
from vital.utils.decorators import auto_move_data
from vital.utils.serialization import resolve_model_checkpoint_path


class TrainEnsemble(SegmentationComputationMixin):
    """Example running commands.

    - Using multiple checkpoints from same run: (NOT WORKING YET)
    `python runner.py +callbacks.model_checkpoint.save_top_k=<Number of ensemble models>`

    - Using multirun with multiple different seeds and data subsets:
    `python runner.py -m system=trainensemble seed=1,2,3...`

    Create ensemble checkpoint:
    - Aggregating from multirun:
    `python crisp_uncertainty/system/ensemble.py --path=<path/to/multirun> \
                                                             --exclude_checkpoint \
                                                             --project_name=echo-segmentation-uncertainty`
    - Aggregating from multiple checkpoints: (NOT WORKING YET)
    `python crisp_uncertainty/system/ensemble.py --path=<path/to/run> \
                                                             --project_name=echo-segmentation-uncertainty`
    """

    def __init__(self, data_split: float = 0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("data_split")

    def on_fit_start(self) -> None:  # noqa: D102
        train_set = self.trainer.datamodule._dataset[Subset.TRAIN]
        indices = random.sample(range(len(train_set)), int(self.hparams.data_split * len(train_set)))
        self.trainer.datamodule._dataset[Subset.TRAIN] = torchdata.Subset(train_set, indices)


class EvalEnsemble(UncertaintyMapEvaluationSystem):
    """Class to evaluate ensembles."""

    def __init__(self, ensemble_ckpt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("ensemble_ckpt")
        self.module_list = torch.load(resolve_model_checkpoint_path(ensemble_ckpt))

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return f"Ensemble-{os.path.basename(self.hparams.ensemble_ckpt)}"

    @auto_move_data
    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        logits = [model(img).detach() for model in self.module_list]

        if logits[0].shape[1] == 1:
            probs = [torch.sigmoid(logits[i]).detach() for i in range(len(logits))]
            probs = torch.stack(probs, dim=-1).cpu().numpy()
            y_hat = probs.mean(-1)
            # uncertainty_map = scipy.stats.entropy(probs, axis=-1)
            # uncertainty_map = 1 - (uncertainty_map - np.min(uncertainty_map)) / np.ptp(uncertainty_map)
            y_hat_prime = np.concatenate([y_hat, 1 - y_hat], axis=1)
            uncertainty_map = scipy.stats.entropy(y_hat_prime, axis=1)
        else:
            probs = [F.softmax(logits[i], dim=1).detach() for i in range(len(logits))]
            y_hat = torch.stack(probs, dim=-1).mean(-1).cpu().numpy()
            uncertainty_map = scipy.stats.entropy(y_hat, axis=1)

        return y_hat, uncertainty_map


if __name__ == "__main__":
    import argparse
    import pprint

    import dotenv
    from comet_ml import Experiment
    from vital.modules.segmentation.enet import Enet
    from vital.modules.segmentation.unet import UNet

    dotenv.load_dotenv(override=True)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--model", default="enet", type=str, choices=["enet", "unet"])
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--project_name", default=None, type=str)
    parser.add_argument("--exclude_checkpoint", action="store_true")
    parser.add_argument("--tags", nargs="*", default=None, type=str)
    params = parser.parse_args()

    paths = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(params.path)
        for f in filenames
        if os.path.splitext(f)[1] == ".ckpt" and ("checkpoint" not in dp if params.exclude_checkpoint else True)
    ]

    print("Aggregating weights from: ")
    pprint.pprint(paths)

    model_list = nn.ModuleList()

    data_params = DataParameters(in_shape=(1, 256, 256), out_shape=(4, 256, 256), labels=None)
    if params.model == "unet":
        module = UNet((1, 256, 256), (params.num_classes, 256, 256))
    elif params.module == "enet":
        module = Enet((1, 256, 256), (params.num_classes, 256, 256))
    else:
        raise ValueError("Unknown model")

    for path in paths:
        model = TrainEnsemble.load_from_checkpoint(path, module=module, data_params=data_params).module
        model_list.append(model)

    torch.save(model_list, "ensemble.ckpt")

    if params.project_name:
        experiment = Experiment(project_name=params.project_name)
        if params.tags:
            experiment.add_tags(params.tags)
        experiment.log_model("model", "ensemble.ckpt")
