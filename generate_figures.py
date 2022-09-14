from comet_ml import APIExperiment
from crisp_uncertainty.evaluation.datasetevaluator import DiceErrorCorrelation
from crisp_uncertainty.evaluation.uncertainty.calibration import PixelCalibration, SampleCalibration
from crisp_uncertainty.evaluation.uncertainty.correlation import Correlation
from crisp_uncertainty.evaluation.uncertainty.distribution import Distribution
from crisp_uncertainty.evaluation.uncertainty.successerrorhistogram import SuccessErrorHist
from crisp_uncertainty.evaluation.uncertainty.vis import UncertaintyVisualization

if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from pathlib import Path

    import dotenv
    from scripts.comet_grouped_plots import get_workspace_experiment_keys

    dotenv.load_dotenv(override=True)

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
    parser.add_argument("--out_dir", default=None, type=Path, help="Output directory where to save the figures")
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

    if args.out_dir is None:
        args.out_dir = Path("+".join(args.include_tag))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(experiment_keys)

    evaluators_classes = [
        Correlation,
        PixelCalibration,
        SampleCalibration,
        Distribution,
        DiceErrorCorrelation,
        SuccessErrorHist,
        UncertaintyVisualization,
    ]
    relevant_files = [c.SAVED_FILES for c in evaluators_classes]
    relevant_files = [item for sublist in relevant_files for item in sublist]
    ids = []
    for experiment_key in experiment_keys:
        exp = APIExperiment(previous_experiment=experiment_key)
        id = exp.get_parameters_summary("id")["valueCurrent"]
        ids.append(id)
        (args.out_dir / id).mkdir(parents=True, exist_ok=True)
        for asset in exp.get_asset_list():
            if os.path.basename(asset["fileName"]) in relevant_files:
                print(f"Exp: {id}: Downloading {asset['fileName']}")
                content = exp.get_asset(asset["assetId"])
                with open(args.out_dir / id / os.path.basename(asset["fileName"]), "wb") as file:
                    file.write(content)

    for evaluator in evaluators_classes:
        evaluator.export_results(ids, args.out_dir)
