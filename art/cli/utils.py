import subprocess
from enum import Enum


class ProjectType(Enum):
    """
    Enum for project types
    """

    CLASSIFICATION = 1
    REGRESSION = 2
    CLUSTERING = 3
    TIME_SERIES = 4


def get_git_user_info():
    """
    Retrieve the git user's username and email.

    Returns:
        Tuple[str, str]: A tuple containing the git username and email.
                         Returns (None, None) if retrieval fails.

    Raises:
        subprocess.CalledProcessError: If the git command fails.
    """
    try:
        username = (
            subprocess.check_output(["git", "config", "--get", "user.name"])
            .decode()
            .strip()
        )
        email = (
            subprocess.check_output(["git", "config", "--get", "user.email"])
            .decode()
            .strip()
        )
        return username, email
    except subprocess.CalledProcessError:
        return None, None
