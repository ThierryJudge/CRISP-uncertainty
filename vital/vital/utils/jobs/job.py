import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, TypedDict

logger = logging.getLogger(__name__)

SbatchCommands = Dict[str, Any]


class SetupOptions(TypedDict):
    """Collection of configuration options describing how to setup scripts to run generic jobs.

    Args:
        commands: Custom commands to run before the main script.
        script: Path of the main script to run for the job.
    """

    script: str
    commands: List[str]


class Job(ABC):
    """Class that can generate job scripts and run them."""

    RUN_CMD: str

    def __init__(
        self,
        job_name: str,
        save_dir: Path,
        setup_options: SetupOptions,
        script_args: str = "",
        enable_log_out: bool = False,
        enable_log_err: bool = False,
    ):
        """Initializes class instance.

        Args:
            job_name: Name of the job registered to SLURM.
            save_dir: Path of the directory where to save the generated script and the optional SLURM logs produced.
            setup_options: Collection of configuration options describing how to setup the job to run inside the script.
            script_args: String of arguments to pass to the main script.
            enable_log_out: Whether to enable a dedicated log for the standard output of the script.
                Output log saved to `save_dir`/'output.out' if enabled.
            enable_log_err: Whether to enable a dedicated log for the error output of the script.
                Error log saved to `save_dir`/'output.err' if enabled.
        """
        self._job_name = job_name
        self._setup_options = setup_options
        self._script_args = script_args
        self._enable_log_out = enable_log_out
        self._enable_log_err = enable_log_err
        # Ensure references to user home directory are expanded when reading/writing the job script
        self._save_dir = Path(os.path.expandvars(save_dir))
        self._job_script_path = self._save_dir / "job_script.sh"

    def write_script(self) -> None:
        """Writes the job's script to disk.

        This method is mainly expected to be used for testing, i.e. to check the generated script before launching it.
        """
        logger.info(f"Writing {self._job_name} job's script...")
        self._job_script_path.parent.mkdir(parents=True, exist_ok=True)
        self._job_script_path.write_text(self._build_script_str())

    def submit(self) -> bool:
        """Writes the job's script to disk and launches the job using `self.RUN_CMD`.

        Returns:
            Whether the job was submitted successfully.
        """
        self.write_script()

        # run script to launch job
        logger.info("Launching job...")
        result = subprocess.run(f"{self.RUN_CMD} {self._job_script_path}", shell=True)
        if submit_success := result.returncode == 0:
            logger.info(f"Launched job {self._job_script_path} \n")
        else:
            logger.warning("Launch failed... \n")
        return submit_success

    @abstractmethod
    def _build_script_str(self) -> str:
        """Generates the string of the job's commands from the parameters describing the job.

        Returns:
            Content of the script that will run the job.
        """
