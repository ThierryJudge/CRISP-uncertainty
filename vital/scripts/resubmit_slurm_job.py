import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_job_dir(job_script_path: Path) -> None:
    for child in job_script_path.parent.iterdir():
        try:
            if child.is_file() and child != job_script_path:
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            logger.warning("Failed to delete %s. Reason: %s" % (child, e))


def _submit_job(job_script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(f"sbatch {job_script_path}", shell=True)


if __name__ == "__main__":
    """Run the script."""
    from argparse import ArgumentParser

    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("job_script", type=Path, nargs="*", help="Path to the script of the job to submit again")
    parser.add_argument(
        "--skip_cleanup",
        dest="cleanup",
        action="store_false",
        help="Whether to skip removing output from the previous submission (except the job script itself) from the job "
        "script's directory",
    )
    args = parser.parse_args()

    for job_script in args.job_script:
        if args.cleanup:
            _clean_job_dir(job_script)

        result = _submit_job(job_script)

        if submit_success := result.returncode == 0:
            logger.info(f"Successfully re-submitted job {job_script} \n")
        else:
            logger.warning(f"Failed to re-submit job {job_script} \n")
