import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from comet_ml import get_config
from comet_ml.api import API, APIExperiment
from tqdm import tqdm

from vital.utils.format.native import filter_excluded
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair

logger = logging.getLogger(__name__)


def get_workspace_experiment_keys(
    include_tags: Sequence[str] = None,
    exclude_tags: Sequence[str] = None,
    to_exclude: Sequence[Union[str, Path]] = None,
) -> List[str]:
    """Retrieves the keys of all the experiments in the workspace/project defined in the `.comet.config` file.

    Args:
        include_tags: Tags that experiments should have to be listed.
        exclude_tags: Tag that experiments should NOTE have to be listed.
        to_exclude: Individual experiment keys, or files listing multiple experiment keys, of specific experiments to
            exclude.

    Returns:
        Keys of some experiments from the workspace/project.
    """
    exclude_tags = exclude_tags if exclude_tags else []
    include_by_default = include_tags is None

    # Get all the experiments in workspace
    config = get_config()
    workspace, project = config["comet.workspace"], config["comet.project_name"]
    workspace_experiments = API().get(workspace, project)

    def experiment_has_tag(experiment: APIExperiment, tag: str) -> bool:
        return any(tag == experiment_tag for experiment_tag in experiment.get_tags())

    # Only keep experiments with the requested metadata
    experiment_keys = []
    for experiment in workspace_experiments:
        any_include_tag = include_by_default or any(experiment_has_tag(experiment, tag) for tag in include_tags)
        any_exclude_tag = any(experiment_has_tag(experiment, tag) for tag in exclude_tags)
        if any_include_tag and not any_exclude_tag:
            experiment_keys.append(experiment.id)

    # Filter out specifically excluded experiments
    experiment_keys = filter_excluded(experiment_keys, to_exclude=to_exclude if to_exclude is not None else [])

    logger.info(
        f"Including {len(experiment_keys)} out of {len(workspace_experiments)} experiments "
        f"from project '{project}' in workspace '{workspace}'"
    )
    return experiment_keys


def get_experiment_available_metrics(experiment: APIExperiment) -> List[str]:
    """Lists the metrics that were logged for a specific experiment.

    Args:
        experiment: Experiment object.

    Returns:
        Name of the metrics logged for the provided experiment.
    """
    return [metric_summary["name"] for metric_summary in experiment.get_metrics_summary()]


def get_experiments_data(experiment_keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Retrieves sampled metrics for experiments by querying the Comet API.

    Args:
        experiment_keys: Keys of the experiments to plot.
        metrics: Metrics for which retrieve the sampled data.

    Returns:
        Sampled metrics from all experiments.
    """
    # Initialize the data structure that will contain all of the experiments' data
    experiments_data = []

    desc_template = "Fetching data from experiment {}"
    pbar = tqdm(experiment_keys)
    for experiment_key in pbar:
        pbar.set_description(desc=desc_template.format(experiment_key))

        # Fetch the current experiment's metadata
        exp = APIExperiment(previous_experiment=experiment_key)
        exp_avail_metrics = get_experiment_available_metrics(exp)
        exp_params = {param["name"]: param["valueCurrent"] for param in exp.get_parameters_summary()}

        # Collect the experiment's data
        for metric_name in (metric_name for metric_name in metrics if metric_name in exp_avail_metrics):
            exp_metric_entries = exp.get_metrics(metric_name)

            # Add the experiment's parameters to each metric entry,
            # to be able to filter data based on experiment parameters
            for exp_metric_entry in exp_metric_entries:
                exp_metric_entry.update(exp_params)

            experiments_data.extend(exp_metric_entries)

    # Organize the collected data in a dataframe, casting types as necessary
    experiments_data = pd.DataFrame(experiments_data)
    experiments_data.metricValue = experiments_data.metricValue.astype(float)

    return experiments_data


def plot_mean_std_curve(
    experiments_data: pd.DataFrame, metric: str, group_by: str, output_dir: Path, scale: str = None
) -> None:
    """Plots the mean and std curve for arbitrary groups of experiments.

    References:
        - Documentation of the supported scales for matplotlib's `Axes`:
          https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.set_yscale.html

    Args:
        experiments_data: Sampled metrics from all experiments.
        metric: Metric for which to plot each group's curve.
        group_by: Hyperparameter by which to group experiments.
        output_dir: Output directory where to save the figures.
        scale: Scale to use when plotting the metric's values.
            Valid scales are those supported by matplotlib (link in the refs).
    """
    # Ensure that matplotlib is using 'agg' backend
    # to avoid possible 'Could not connect to any X display' errors
    # when no X server is available, e.g. in remote terminal
    plt.switch_backend("agg")

    plot_title = f"{metric} w.r.t. {group_by}"
    logger.info(f"Generating {plot_title} plot ...")

    # Filter the experiments' data to only include data for the metric of interest
    data = experiments_data.loc[experiments_data.metricName == metric]

    with sns.axes_style("darkgrid"):
        ax = sns.lineplot(data=data, x="epoch", y="metricValue", hue=group_by)
        ax.set_title(plot_title)
        ax.set_ylabel(metric)
        if scale is not None:
            ax.set(yscale=scale)

    output_file = output_dir / f"{metric}_{group_by}.png"
    plt.savefig(output_file)
    plt.close()  # Close the figure in case the function is called multiple times

    logger.info(f"Saved plot at : {output_file} ... \n")


def main():
    """Run the script."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_key",
        type=str,
        nargs="+",
        help="Key of the experiment to plot. If no experiment is provided, defaults to plotting for all the "
        "experiments in the workspace. NOTE: Manually specifying the experiment keys disables all other filters "
        "(e.g. `--include_tag`, `--exclude_experiment`, etc.)",
    )
    parser.add_argument(
        "--include_tag",
        type=str,
        nargs="+",
        help="Tag that experiments should have to be included in the plots",
    )
    parser.add_argument(
        "--exclude_tag",
        type=str,
        nargs="+",
        help="Tag that experiments should NOT have to be included in the plots",
    )
    parser.add_argument(
        "--exclude_experiment",
        type=str,
        nargs="+",
        help="Key of the experiment to plot, or path to a file listing key of experiments to exclude from the plots",
    )
    parser.add_argument(
        "--metric", type=str, nargs="+", help="Metric for which to plot each group's curve", required=True
    )
    parser.add_argument(
        "--scale",
        action=StoreDictKeyPair,
        default=dict(),
        metavar="METRIC1=SCALE1,METRIC2=SCALE2...",
        help="Scale to use for a metric. By default, metrics are plotted on a linear scale. Here, you can specify "
        "custom scales for each metric.",
    )
    parser.add_argument("--group_by", type=str, help="Hyperparameter by which to group experiments", required=True)
    parser.add_argument("--out_dir", type=Path, help="Output directory where to save the figures", required=True)
    args = parser.parse_args()

    # Cast to path excluded experiments arguments that are valid file paths
    excluded_experiments = (
        [
            Path(exclude_item) if os.path.isfile(exclude_item) else exclude_item
            for exclude_item in args.exclude_experiment
        ]
        if args.exclude_experiment
        else None
    )

    # Determine the experiments to include in the plots
    experiment_keys = args.experiment_key
    if not experiment_keys:
        experiment_keys = get_workspace_experiment_keys(
            include_tags=args.include_tag, exclude_tags=args.exclude_tag, to_exclude=excluded_experiments
        )

    if not (num_experiments := len(experiment_keys)) > 1:
        raise ValueError(
            f"Cannot generate plots for only one experiment. Please provide at least "
            f"{2 - num_experiments} other experiment(s) for which to plot curves."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    experiments_data = get_experiments_data(experiment_keys, args.metric)
    for metric in args.metric:
        plot_mean_std_curve(experiments_data, metric, args.group_by, args.out_dir, scale=args.scale.get(metric))


if __name__ == "__main__":
    main()
