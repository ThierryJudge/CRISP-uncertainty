import logging
import re
import shutil
from pathlib import Path
from typing import Union

import comet_ml
from packaging.version import InvalidVersion, Version

from vital import get_vital_home

logger = logging.getLogger(__name__)


def fix_registry_name(registry_name: str) -> str:
    """Changes registry name to replace characters that COMET ML does not accept in the registry name.

    Note: Comet ML rules (Might need to be updated)
        - All special characters are replaced with dash (ex. camus_unet -> camus-unet, camus!unet -> camus-unet)
        - All non-letter and non-number characters at the end of the name are deleted.
        - All letters are lower-case.

    Args:
        registry_name: Name of the registered model to update.

    Returns:
        Updated registry name
    """
    registry_name = re.sub("[^a-zA-Z0-9]+", "-", registry_name)  # Replace all special characters with `-`
    registry_name = re.sub(r"^\W+|\W+$", "", registry_name)  # Remove all special characters at end of name

    return registry_name.lower()


def resolve_model_checkpoint_path(checkpoint: Union[str, Path]) -> Path:
    """Resolves a local path or a Comet model registry query to a local path on the machine.

    Notes:
        - If the `ckpt` is to be downloaded off of a Comet model registry, your Comet API key needs to be set in one of
          Comet's expected locations: https://www.comet.ml/docs/python-sdk/advanced/#non-interactive-setup

    Args:
        checkpoint: Location of the checkpoint. This can be either a local path, or the fields of a query to a Comet
            model registry. Examples of different queries:
                - For the latest version of the model: 'my_workspace/my_model'
                - Using a specific version/stage: 'my_workspace/my_model/0.1.0' or 'my_workspace/my_model/prod'

    Returns:
        Path to the model's checkpoint file on the local computer. This can either be the `ckpt` already provided, if it
        was already local, or the location where the checkpoint was downloaded, if it pointed to a Comet registry model.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".ckpt":
        local_ckpt_path = checkpoint
    else:
        try:
            comet_api = comet_ml.api.API()
        except ValueError:
            raise RuntimeError(
                f"The format of the checkpoint '{checkpoint}' indicates you want to download a model from a Comet "
                f"model registry, but Comet couldn't find an API key. Either switch to providing a local checkpoint "
                f"path, or set your Comet API key in one of Comet's expected locations."
            )

        # Parse the provided checkpoint path as a query for a Comet model registry
        version_or_stage, version, stage = None, None, None
        if len(checkpoint.parts) == 2:
            workspace, registry_name = checkpoint.parts
        elif len(checkpoint.parts) == 3:
            workspace, registry_name, version_or_stage = checkpoint.parts
            try:
                Version(version_or_stage)  # Will fail if `version_or_stage` cannot be parsed as a version
                version = version_or_stage
            except InvalidVersion:
                stage = version_or_stage
        else:
            raise ValueError(f"Failed to interpret checkpoint '{checkpoint}' as a query for a Comet model registry.")

        registry_name = fix_registry_name(registry_name)

        # If neither version nor stage were provided, use latest version available
        if not version_or_stage:
            version_or_stage = version = comet_api.get_registry_model_versions(workspace, registry_name)[-1]

        # Determine where to download the checkpoint locally
        cache_dir = get_vital_home()
        model_cached_path = cache_dir / workspace / registry_name / version_or_stage

        # When using stage, delete cached versions and force re-downloading the registry model,
        # because stage tags can be changed
        if stage:
            shutil.rmtree(model_cached_path, ignore_errors=True)

        # Download model if not already cached
        if not model_cached_path.exists():
            comet_api.download_registry_model(
                workspace, registry_name, version=version, stage=stage, output_path=str(model_cached_path)
            )
        else:
            logger.info(
                f"Using cached registry model {registry_name}, version {version} from workspace {workspace} "
                f"located in '{model_cached_path}'."
            )

        # Extract the path of the checkpoint file on the local machine
        ckpt_files = list(model_cached_path.glob("*.ckpt"))
        if len(ckpt_files) != 1:
            raise RuntimeError(
                f"Expected the Comet model to contain a single '*.ckpt' file, but there were {len(ckpt_files)} "
                f"'*.ckpt' file(s): {ckpt_files}. Either edit the content of the Comet model, or use a different model."
            )
        local_ckpt_path = ckpt_files[0]

    return local_ckpt_path
