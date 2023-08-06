import typer
import os
import shutil
from git import Repo
import time

URL = "https://github.com/SebChw/art_template.git"

app = typer.Typer()


@app.command()
def create_project(project_name: str, keep_as_repo: bool = False) -> None:
    try:
        _ = Repo.clone_from(URL, project_name, branch="main", depth=1)
    except Exception as e:
        print("Error while cloning the template's repository:", str(e))
        return

    try:
        time.sleep(1)
        git_folder = os.path.join(project_name, ".git")
        shutil.rmtree(git_folder, ignore_errors=True)
    except Exception as e:
        print("System error while deleting .git directory:", str(e))
        return

    print(f"Project created in {project_name}/")


@app.command()
def run_training(project_name: str, args):
    pass


@app.command()
def run_evaluation(project_name: str, args):
    pass


@app.command()
def run_checks(project_name: str, args):
    pass


@app.command()
def do_step(project_name: str, step_name: str, args):
    pass


@app.command()
def add_component():
    # One can add Mlbaseline etc. to the codebase
    # python -m art add ml_baseline -> in local folder new file is created.
    pass


if __name__ == "__main__":
    app()
