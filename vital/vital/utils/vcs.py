import subprocess

UNKNOWN_GIT_VERSION: str = "Unknown"
"""Literal string to use when the Git revision of the code cannot be determined."""


def get_git_revision_hash(short: bool = False) -> str:
    """Obtains the hash of the Git revision of the project's code.

    Notes:
        - If the Git revision cannot be found, for whatever reason (e.g. the project is not a Git repository), the
          function won't crash, but will rather return a literal: module level variable ``UNKNOWN_GIT_VERSION``.

    Args:
        short: If ``True``, use the short version of the hash; otherwise, use the default long version.

    Returns:
        Hash of the Git revision of the project's code, if it can be determined; otherwise, the literal value of
        ``UNKNOWN_GIT_VERSION``.
    """
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd.insert(-1, "--short")

    try:
        git_revision = subprocess.check_output(cmd).strip().decode("ascii")
    except subprocess.CalledProcessError:
        git_revision = UNKNOWN_GIT_VERSION

    return git_revision
