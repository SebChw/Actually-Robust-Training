import os
import sys
import shutil
from git import Repo
import time

URL = "https://github.com/kordc/art-basetemplate.git"

def create_project(destination_folder: str) -> None:
    try:
        #TODO add different branches for different templates
        _ = Repo.clone_from(URL, destination_folder, branch="base", depth=1)
    except Exception as e:
        print("Error while cloning the template's repository:", str(e))
        return

    try:
        time.sleep(1)
        git_folder = os.path.join(destination_folder, ".git")
        shutil.rmtree(git_folder, ignore_errors=True)
    except Exception as e:
        print("System error while deleting .git directory:", str(e))
        return

    print(f"Project created in {destination_folder}")

def is_valid_path(path : str) -> bool:
    if path[0] in ["-", "--"]:
        return False
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2 or not is_valid_path(sys.argv[1]):
        print("Usage: python -m art.cli.create_project <destination_folder>")
    else:
        destination_folder = sys.argv[1]
        create_project(destination_folder)
