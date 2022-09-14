if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from pathlib import Path

    import dotenv
    from scripts.comet_grouped_plots import get_experiments_data, get_workspace_experiment_keys

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
    parser.add_argument(
        "--out_dir", default=Path.cwd() / "test", type=Path, help="Output directory where to save the figures"
    )
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

    # Determine the experiments to include in the plotsN
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

    uncertainty_metrics = [
        "test_dice",
        "test_Correlation",
        "test_overlap",
        "test_PixelCalibration_ece",
        "test_SampleCalibration_ece",
    ]

    uncertainty_experiments_data = get_experiments_data(experiment_keys, uncertainty_metrics)
    uncertainty_experiments_data = uncertainty_experiments_data[["id", "metricName", "metricValue"]]
    df = uncertainty_experiments_data.pivot("id", "metricName", "metricValue")

    # df = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)), columns=list('ABCD'))

    def bold_extreme_values(data, data_max=-1):  # # noqa: D103
        if data == data_max:
            return "\\textbf{%.4f}" % data
        else:
            return f"{data:.4f}"

    print(df)

    for col in df.columns.get_level_values(0).unique():
        df[col] = df[col].apply(lambda data: bold_extreme_values(data, data_max=df[col].max()))

    print(df.to_latex(escape=True))
