import os
from pathlib import Path
from typing import List

import numpy as np
from crisp_uncertainty.evaluation.uncertainty.mutual_information import UncertaintyErrorMutualInfo
from mpl_toolkits.axes_grid1 import make_axes_locatable

from crisp_uncertainty.evaluation.data_struct import PatientResult
from crisp_uncertainty.evaluation.patientevaluator import PatientEvaluator
from crisp_uncertainty.evaluation.uncertainty.overlap import UncertaintyErrorOverlap
from crisp_uncertainty.utils.numpy import prob_to_categorical
from matplotlib import pyplot as plt, gridspec, colors


class UncertaintyVisualization(PatientEvaluator):
    """Evaluator to generate uncertainty visualisations."""

    VIS_FILE_NAME = "UncertaintyVisualization.npy"
    SAVED_FILES = [VIS_FILE_NAME]

    def __init__(self, uncertainty_threshold: float = 0.05, nb_upload=50, nb_figures=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold
        self.nb_figures = nb_figures
        self.nb_upload = nb_upload
        self.count = 0

        os.mkdir("figures")

    def __call__(self, results: List[PatientResult]):
        """Generates uncertainty visualisations.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            No metrics.
        """
        saved_data = {}
        for patient in results:
            if self.count < self.nb_figures:
                for view, data in patient.views.items():
                    for instant, i in data.instants.items():
                        pred = prob_to_categorical(data.pred[i])
                        error = 1 * ~np.equal(pred, data.gt[i])
                        uncertainty = (data.uncertainty_map[i] > self.uncertainty_threshold).astype(int)

                        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
                        f.set_figheight(9)
                        f.set_figwidth(16)
                        plt.suptitle(f"{patient.id}_{view}_{instant}")
                        ax1.axis("off"), ax2.axis("off"), ax3.axis("off"), ax4.axis("off"),  # ax5.axis("off")
                        ax5.set_xticks([])
                        ax5.set_yticks([])
                        ax1.imshow(data.gt[i].squeeze(), cmap='gray')
                        ax1.set_title("GT")
                        ax2.imshow(pred.squeeze(), cmap='gray')
                        ax2.set_title("Pred")

                        ax3.set_title("Error")
                        ax3.imshow(error.squeeze(), cmap='gray')

                        im = ax4.imshow(data.uncertainty_map[i].squeeze(), cmap='gray')
                        ax4.set_title(
                            f"Unc. (min: {np.min(data.uncertainty_map[i]):.3f}, "
                            f"max: {np.max(data.uncertainty_map[i]):.3f})"
                        )
                        divider = make_axes_locatable(ax4)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        f.colorbar(im, cax=cax)

                        ax5.imshow(uncertainty.squeeze(), cmap='gray')

                        ax5.yaxis.set_label_position("right")

                        ax5.set_title("Thresh. Unc.")
                        o = UncertaintyErrorOverlap.compute_overlap(1 * ~np.equal(pred, data.gt[i]).squeeze(),
                                                                    data.uncertainty_map[i], self.uncertainty_threshold)
                        mi = UncertaintyErrorMutualInfo.compute_mi(1 * ~np.equal(pred, data.gt[i]).squeeze(),
                                                                   data.uncertainty_map[i])
                        ax5.set_ylabel(f"{round(o, 3)}\n{round(mi, 5)}", rotation=0, labelpad=25)
                        ax5.yaxis.set_label_position("right")

                        if self.count < self.nb_upload:
                            patient_data = {
                                "img": data.img[i],
                                "pred": pred,
                                "gt": data.gt[i],
                                "unc": data.uncertainty_map[i],
                                "unc_thresh": uncertainty,
                            }
                            saved_data[f"{patient.id}_{view}_{instant}"] = patient_data
                            plt.savefig(self.upload_dir / f"{patient.id}_{view}_{instant}.png", dpi=100)
                        else:
                            plt.savefig(f"figures/{patient.id}_{view}_{instant}.png", dpi=100)
                        plt.close()
                        self.count += 1

        np.save(self.upload_dir / f"{self.__class__.__name__}.npy", saved_data)

        return {}, {}

    @classmethod
    def export_results(cls, experiment_names: List[str], data_dir: Path, **kwargs):
        """Aggregates and exports results for evaluator.

        Args:
            experiment_names: List of experiment names.
            data_dir: Path to the downloaded data
        """
        num_fig = 25

        column_names = ['gt', 'pred', 'error', 'unc', 'unc_thresh']
        for i in range(num_fig):
            f = plt.figure()
            # f.set_figheight(5)
            f.set_figwidth(8)
            gs = gridspec.GridSpec(len(experiment_names), len(column_names))
            gs.update(wspace=0.25, hspace=0.05)

            for j, exp in enumerate(experiment_names):
                data = np.load(data_dir / exp / cls.VIS_FILE_NAME, allow_pickle=True)[()]
                patient_name = list(data.keys())[i]
                data = data[patient_name]
                plt.suptitle(patient_name)
                for k, col in enumerate(column_names):
                    ax = plt.subplot(gs[j, k])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Set the column name if this is the top row.
                    if j == 0:
                        ax.set_title(col)

                    # Set the row name if this is the first column.
                    if k == 0:
                        ax.set_ylabel(exp.split('-')[0])

                    if col == 'error':
                        img_data = 1 * ~np.equal(data['pred'], data['gt']).squeeze()
                    else:
                        img_data = data[col]

                    im = ax.imshow(img_data.squeeze())

                    if k == len(column_names) - 1:
                        o = UncertaintyErrorOverlap.compute_overlap(1 * ~np.equal(data['pred'], data['gt']).squeeze(),
                                                                    data['unc_thresh'], 0)
                        mi = UncertaintyErrorMutualInfo.compute_mi(1 * ~np.equal(data['pred'], data['gt']).squeeze(),
                                                                   data['unc'])
                        ax.set_ylabel(f"{round(o, 3)}\n{round(mi, 5)}", rotation=0, labelpad=25)
                        ax.yaxis.set_label_position("right")

                    if col == 'unc':
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        f.colorbar(im, cax=cax, boundaries=np.linspace(0, 1, 100))

            # print(patient_name, plt.gcf().get_size_inches())
            plt.savefig(data_dir / f"{patient_name}.png")
            plt.close()

    # @classmethod
    # def export_results(cls, experiment_names: List[str], data_dir: Path, **kwargs):
    #     """Aggregates and exports results for evaluator.
    #
    #     Args:
    #         experiment_names: List of experiment names.
    #         data_dir: Path to the downloaded data
    #     """
    #     num_fig = 3
    #
    #     f = plt.figure()
    #     f.set_figheight(2 * num_fig)
    #     f.set_figwidth(len(experiment_names) + 1)
    #
    #     gs = gridspec.GridSpec(2 * num_fig, len(experiment_names), width_ratios=np.ones(len(experiment_names)).tolist())
    #     gs.update(wspace=0.025, hspace=0.05)
    #
    #     # aspect = 'auto'
    #     aspect = None
    #
    #     for i in range(num_fig):
    #         for j, exp in enumerate(experiment_names):
    #             data = np.load(data_dir / exp / cls.VIS_FILE_NAME, allow_pickle=True)[()]
    #             name = list(data.keys())[i + 3]
    #             data = data[name]
    #
    #             if i == 0:
    #                 ax = plt.subplot(gs[2 * i, j])
    #                 ax.set_title(exp.split('-')[0])
    #
    #             # if j == 0:
    #             #     ax = plt.subplot(gs[2 * i, j])
    #             #     if i % 2 == 0:
    #             #         print("ERROR")
    #             #         ax.set_ylabel('Error', rotation=0)
    #             #         # ax.yaxis.set_label_position("right")
    #             #     else:
    #             #         print("ERROR")
    #             #         ax.set_ylabel('Unc.', rotation=0)
    #             #         # ax.yaxis.set_label_position("right")
    #
    #             # if j == 0:
    #             #     ax = plt.subplot(gs[2 * i, 0])
    #             #     ax.imshow(data['img'].squeeze(), aspect=aspect)
    #             #     # axes[2*i, 0].axis('off')
    #             #     ax.set_xticks([])
    #             #     ax.set_yticks([])
    #             #     ax.axis('off')
    #             #     ax.set_ylabel(name)
    #
    #             ax = plt.subplot(gs[2 * i, j])
    #             # ax.imshow(data['pred'].squeeze(), aspect=aspect)
    #             ax.imshow(1 * ~np.equal(data['pred'], data['gt']).squeeze(), aspect=aspect)
    #             # ax.axis('off')
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             if j == 0:
    #                 ax.set_ylabel('Error', rotation=0, labelpad=25)
    #
    #             ax = plt.subplot(gs[2 * i + 1, j])
    #             im = ax.imshow(data['unc'].squeeze(), aspect=aspect)
    #
    #             divider = make_axes_locatable(ax)
    #             cax = divider.append_axes("right", size="5%", pad=0.05)
    #             f.colorbar(im, cax=cax)
    #
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             # ax.axis('off')
    #             if j == 0:
    #                 ax.set_ylabel('Unc.', rotation=0, labelpad=50)
    #
    #     f.tight_layout()
    #     plt.savefig(data_dir / "Vis.png")
