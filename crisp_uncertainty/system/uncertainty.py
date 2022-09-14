from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything

from crisp_uncertainty.evaluation.data_struct import PatientResult, ViewResult
from crisp_uncertainty.evaluation.datasetevaluator import DiceErrorCorrelation, ThresholdedDice, \
    ThresholdErrorOverlap
from crisp_uncertainty.evaluation.segmentation.metrics import SegmentationMetrics
from crisp_uncertainty.evaluation.uncertainty.calibration import PixelCalibration, SampleCalibration, \
    PatientCalibration
from crisp_uncertainty.evaluation.uncertainty.correlation import Correlation
from crisp_uncertainty.evaluation.uncertainty.distribution import Distribution
from crisp_uncertainty.evaluation.uncertainty.stats import Stats
from crisp_uncertainty.evaluation.uncertainty.mutual_information import UncertaintyErrorMutualInfo
from crisp_uncertainty.evaluation.uncertainty.overlap import UncertaintyErrorOverlap
from crisp_uncertainty.evaluation.uncertainty.successerrorhistogram import SuccessErrorHist
from crisp_uncertainty.evaluation.uncertainty.vis import UncertaintyVisualization
from crisp_uncertainty.utils.metrics import Dice
from crisp_uncertainty.utils.numpy import prob_to_categorical
from crisp_uncertainty.utils.reductions import available_reductions
from pytorch_lightning.loggers import CometLogger
from torch import Tensor
from tqdm import tqdm
from vital.data.camus.config import Label
from vital.data.camus.data_struct import PatientData, ViewData
from vital.data.config import Tags
from vital.systems.system import SystemEvaluationMixin
from vital.utils.format.native import prefix
from matplotlib import pyplot as plt

class UncertaintyEvaluationSystem(SystemEvaluationMixin):
    """Mixin for handling the evaluation phase for uncertainty and segmentation."""

    UPLOAD_DIR_NAME = "upload"

    def __init__(self, uncertainty_threshold: float = -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold

    def on_test_start(self) -> None:  # noqa: D102
        self.upload_dir = Path.cwd() / self.UPLOAD_DIR_NAME
        self.upload_dir.mkdir(parents=False, exist_ok=False)

        if isinstance(self.trainer.logger, CometLogger):
            name = self.get_name()
            if name is not None:
                self.trainer.logger.experiment.set_name(name)
                self.trainer.logger.log_hyperparams(Namespace(**{"id": name}))

    def get_name(self):
        """Get name of current method for saving.

        Returns:
            name of method
        """
        return None

    def test_step(self, batch: PatientData, batch_idx: int):
        """Performs test-time inference for a patient and saves the results to an HDF5 file."""
        return self.compute_patient_uncertainty(batch)

    def compute_patient_uncertainty(self, batch: PatientData) -> PatientResult:
        """Computes the uncertainty for one patient.

        Args:
            batch: data for one patient including image, groundtruth and other information.

        Returns:
            PatientResult struct containing uncertainty prediction
        """
        patient_res = PatientResult(id=batch.id)
        for view, data in batch.views.items():
            patient_res.views[view] = self.compute_view_uncertainty(view, data)
        return patient_res

    def compute_view_uncertainty(self, view: str, data: ViewData) -> ViewResult:
        """Computes the uncertainty for one view.

        Args:
            batch: data for one view including image, groundtruth and other information.

        Returns:
            ViewResult struct containing uncertainty prediction
        """
        raise NotImplementedError

    def test_epoch_end(self, outputs: List[PatientResult]) -> None:
        """Aggregates results.

        Args:
            outputs: List of results for every patient.
        """
        patient_evaluators = [
            SegmentationMetrics(labels=self.hparams.data_params.labels, upload_dir=self.upload_dir),
            Correlation(Dice(labels=self.hparams.data_params.labels), upload_dir=self.upload_dir),
            PixelCalibration(upload_dir=self.upload_dir),
            SampleCalibration(accuracy_fn=Dice(labels=self.hparams.data_params.labels), upload_dir=self.upload_dir),
            UncertaintyErrorOverlap(uncertainty_threshold=self.uncertainty_threshold, upload_dir=self.upload_dir),
            Stats(uncertainty_threshold=self.uncertainty_threshold, upload_dir=self.upload_dir),
            Distribution(upload_dir=self.upload_dir),
            UncertaintyVisualization(
                uncertainty_threshold=self.uncertainty_threshold, nb_figures=50, upload_dir=self.upload_dir
            ),
            SuccessErrorHist(upload_dir=self.upload_dir),
            PatientCalibration(upload_dir=self.upload_dir),
            UncertaintyErrorMutualInfo(upload_dir=self.upload_dir)
        ]

        dataset_evaluators = [
            DiceErrorCorrelation(upload_dir=self.upload_dir),
            ThresholdedDice(upload_dir=self.upload_dir),
            ThresholdErrorOverlap()]

        metrics = {}
        patient_metrics = []
        for evaluator in patient_evaluators:
            print(f"Generating results with {evaluator.__class__.__name__}...")
            try:
                evaluator_metrics, evaluator_patient_metrics = evaluator(outputs)
                patient_metrics.append(evaluator_patient_metrics)
                metrics.update(evaluator_metrics)
            except Exception as e:
                print(f"Failed with exception {e}")

        patient_metrics = pd.concat([pd.DataFrame(m).T for m in patient_metrics], axis=1)

        patient_metrics.to_csv(self.upload_dir / "patient_metrics.csv")

        for evaluator in dataset_evaluators:
            print(f"Generating results with {evaluator.__class__.__name__}...")
            try:
                evaluator_metrics = evaluator(patient_metrics)
                metrics.update(evaluator_metrics)
            except Exception as e:
                print(f"Failed with exception {e}")

        if isinstance(self.trainer.logger, CometLogger):
            self.trainer.logger.experiment.log_asset_folder(str(self.upload_dir), log_file_name=False)

        metrics = prefix(metrics, "test_")
        self.log_dict(metrics)


class UncertaintyMapEvaluationSystem(UncertaintyEvaluationSystem):
    """Evaluation system for methods that evaluate uncertainty maps for each sample (one image)."""

    def __init__(self, reduction: str, normalize: bool = False, mask_bg_uncertainty: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("reduction", "normalize", "mask_bg_uncertainty")
        self.reduction = available_reductions[reduction]

    def compute_view_uncertainty(self, view: str, data: ViewData) -> ViewResult:
        """Computes the uncertainty for one view.

        Args:
            view: Name of the view
            data: data for one view including image, groundtruth and other information.

        Returns:
            ViewResult struct containing uncertainty prediction
        """
        pred, uncertainty_map = self.compute_seg_and_uncertainty_map(data.img_proc)

        if self.hparams.normalize:
            min = uncertainty_map.min(axis=0)
            max = uncertainty_map.max(axis=0)
            uncertainty_map = (uncertainty_map - min) / (max - min)

        # if self.hparams.mask_bg_uncertainty:
        #     mask = prob_to_categorical(pred) != Label.BG.value
        #     frame_uncertainties = self.reduction(np.where(mask, uncertainty_map, np.nan))
        # else:
        #     frame_uncertainties = self.reduction(uncertainty_map)

        mask = prob_to_categorical(pred) != Label.BG.value
        # IF mask sum is 0, use prediction sum to avoid inf uncertainty
        frame_uncertainties = np.sum(uncertainty_map.squeeze(), axis=(-2, -1)) / np.sum(mask, axis=(-2, -1))
        # frame_uncertainties = (np.sum(uncertainty_map, axis=(-2, -1)).squeeze() / np.sum(mask, axis=(-2, -1))).crisp(max=1)

        return ViewResult(
            img=data.img_proc.cpu().numpy(),
            gt=data.gt_proc.cpu().numpy(),
            pred=pred,
            uncertainty_map=uncertainty_map,
            frame_uncertainties=frame_uncertainties,
            view_uncertainty=np.mean(frame_uncertainties),
            voxelspacing=data.voxelspacing,
            instants=data.instants,
        )

    def on_test_start(self) -> None:  # noqa: D102
        super().on_test_start()
        # Compute ideal threshold based of validation set.
        if self.uncertainty_threshold == -1:
            print("Finding ideal threshold...")
            self.trainer.datamodule.setup("fit")  # Need to access Validation set
            val_dataloader = self.trainer.datamodule.val_dataloader()
            errors, uncertainties, error_sums = [], [], []

            for _, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Predicting on val set"):
                x, y = data[Tags.img], data[Tags.gt]
                y_hat, unc = self.compute_seg_and_uncertainty_map(x)

                err = ~np.equal(prob_to_categorical(y_hat), y.numpy())
                errors.append(err)
                error_sums.append(err.sum(axis=(1, 2)))
                uncertainties.append(unc)

            errors, uncertainties, error_sums = np.concatenate(errors), np.concatenate(uncertainties), np.concatenate(error_sums)

            all_dices = []
            thresholds = np.arange(0.025, 1, 0.025)
            for thresh in tqdm(thresholds, desc="Finding ideal threshold"):
                dices = []
                for e, u in zip(errors, uncertainties):
                    dices.append(UncertaintyErrorOverlap.compute_overlap(e, u, thresh))
                all_dices.append(np.average(dices, weights=error_sums))

            self.uncertainty_threshold = thresholds[np.argmax(all_dices)]

        self.log("best_uncertainty_threshold", self.uncertainty_threshold)
        seed_everything(0, workers=True)

    def compute_seg_and_uncertainty_map(self, img: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        raise NotImplementedError
