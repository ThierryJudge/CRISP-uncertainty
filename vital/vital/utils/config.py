from configparser import ConfigParser
from pathlib import Path


def read_ini_config(ini_config: Path) -> ConfigParser:
    """Reads all values from an ini configuration file.

    Args:
        ini_config: Path of the ini configuration file to read.

    Returns:
        Two-tiered mapping, with section names as first level keys and value keys as second level keys.
    """
    config = ConfigParser()
    config.read(str(ini_config))
    return config
