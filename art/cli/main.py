import os
import shutil

import typer
from cookiecutter.main import cookiecutter
from github import Github

from art.cli.utils import get_git_user_info

TEMPLATE_URL = "https://github.com/SebChw/art_template.git"

app = typer.Typer()


@app.command()
def create_project(
    project_name: str, keep_as_repo: bool = False, branch: str = "main"
) -> None:
    """
    Create a new project using a specified Cookiecutter template.

    Args:
        project_name (str): The name of the new project.
        keep_as_repo (bool, optional): Whether to keep the `.git` directory. Defaults to False.
        branch (str, optional): The branch of the template to use. Defaults to "main".
    """

    git_username, git_email = get_git_user_info()

    try:
        cookiecutter(
            TEMPLATE_URL,
            no_input=True,
            extra_context={
                "project_name": project_name,
                "author": git_username,
                "email": git_email,
            },  # Pass the project_name to the template,
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

    if typer.confirm("Do you want to create a new repository for this project?"):
        try:
            os.chdir(project_name)
            os.system(f"git init")
            os.system(f"git add .")
            os.system(f'git commit -m "Initial commit"')
        except Exception as e:
            print("Error while creating repository:", str(e))

    if typer.confirm(
        "Do you want to create a new GitHub repository and push the local one to it (it requires GitHub access token)?"
    ):
        try:
            github_token = typer.prompt("Please enter your GitHub access token")
            g = Github(github_token)
            user = g.get_user()
            repo = user.create_repo(project_name)
            current_branch = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
            os.system(f"git remote add origin {repo.clone_url}")
            os.system(f"git push -u origin {current_branch}")
        except Exception as e:
            print("Error while creating GitHub repository:", str(e))


@app.command()
def get_started():
    """Create a project named 'mnist_tutorial' using the 'mnist_tutorial_cookiecutter' branch."""
    create_project(project_name="mnist_tutorial", branch="mnist_tutorial_cookiecutter")


@app.command()
def run_dashboard(experiment_folder: str = "."):
    """Run dashboard for a given experiment folder."""
    os.system(f"python -m art.dashboard.app --exp_path {experiment_folder}")


@app.command()
def bert_transfer_learning_tutorial():
    """Create a project named 'bert_transfer_learning_tutorial' using the 'bert_transfer_learning_tutorial' branch."""
    create_project(
        project_name="bert_transfer_learning_tutorial",
        branch="bert_transfer_learning_tutorial",
    )


if __name__ == "__main__":
    app()
