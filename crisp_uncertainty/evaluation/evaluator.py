from pathlib import Path
from typing import List, Optional


class Evaluator:
    """Generic class to evaluate predictions.

    Args:
        save_dir: path directory to save files locally
        upload_dir: path to directory to save files to be uploaded to Comet ML.
    """

    SAVED_FILES: List[str] = []

    def __init__(self, save_dir: Path = None, upload_dir: Path = None):
        self.save_dir = save_dir
        self.upload_dir = upload_dir

    @classmethod
    def export_results(
        cls, experiment_names: List[str], data_dir: Path, num_rows: Optional[int] = None, num_cols: Optional[int] = None
    ):
        """Aggregates and exports results for evaluator.

        Args:
            experiment_names: List of experiment names.
            data_dir: Path to the downloaded data
            num_rows: Number of rows for subplots.
            num_cols: Number of columns for subplots.
        """
        pass
