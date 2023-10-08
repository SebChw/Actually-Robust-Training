import typer
import os
import shutil
from cookiecutter.main import cookiecutter
from art.cli.utils import get_git_user_info

TEMPLATE_URL = "https://github.com/SebChw/art_template.git"

app = typer.Typer()


@app.command()
def create_project(project_name: str, keep_as_repo: bool = False, branch: str = "main") -> None:
    git_username, git_email = get_git_user_info()

    try:
        cookiecutter(
            TEMPLATE_URL,
            no_input=True,
            extra_context={"project_name": project_name, "author": git_username, "email": git_email},  # Pass the project_name to the template,
            checkout=branch,  # Use the latest version of the template
        )
    except Exception as e:
        print("Error while generating project using Cookiecutter:", str(e))
        return

    if not keep_as_repo:
        try:
            git_folder = os.path.join(project_name, ".git")
            shutil.rmtree(git_folder, ignore_errors=True)
        except Exception as e:
            print("System error while deleting .git directory:", str(e))
            return

    print(f"Project created in {project_name}/")


@app.command()
def get_started():
    create_project(project_name="mnist_tutorial", branch="mnist_tutorial_cookiecutter")


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
