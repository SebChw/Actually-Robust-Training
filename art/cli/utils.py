import subprocess
from enum import Enum


class ProjectType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    CLUSTERING = 3
    TIME_SERIES = 4


def get_git_user_info():
    try:
        username = subprocess.check_output(["git", "config", "--get", "user.name"]).decode().strip()
        email = subprocess.check_output(["git", "config", "--get", "user.email"]).decode().strip()
        return username, email
    except subprocess.CalledProcessError:
        return None, None
