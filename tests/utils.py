import json
import os
import shutil

from cookiecutter.main import cookiecutter


def clean_up(folder_path):
    """
    Removes pointed directory.
    """
    shutil.rmtree(folder_path)


def check_expected_steps_executions(outputs_path, expected_steps_executions):
    for results_file, expected_results in expected_steps_executions.items():
        with open(os.path.join(outputs_path, results_file), "r") as f:
            results = json.load(f)

        for step_run, expected_result in zip(results["runs"], expected_results):
            assert step_run["successful"] == expected_result


def download_tutorial(proj_name: str, branch: str):
    """
    Downloads the tutorial from the GitHub repository using Cookiecutter.
    """
    try:
        cookiecutter(
            "https://github.com/SebChw/ART-Templates.git",
            no_input=True,
            extra_context={
                "project_name": proj_name,
                "author": "test",
                "email": "test",
            },  # Pass the project_name to the template,
            checkout=branch,  # Use the latest version of the template
        )
    except Exception as e:
        print("Error while generating project using Cookiecutter:", str(e))
        raise e
