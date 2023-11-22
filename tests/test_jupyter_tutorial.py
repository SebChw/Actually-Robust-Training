import json
import os
import shutil

import nbformat
from cookiecutter.main import cookiecutter
from nbconvert.preprocessors import ExecutePreprocessor

EXPECTED_STEPS_EXECUTIONS = {
    "art_checkpoints/AlreadyExistingSolutionBaseline_Evaluate Baseline/results.json": [
        True
    ],
    "art_checkpoints/Data analysis/results.json": [True],
    "art_checkpoints/HeuristicBaseline_Evaluate Baseline/results.json": [True],
    "art_checkpoints/MlBaseline_Evaluate Baseline/results.json": [True],
    "art_checkpoints/MNISTModel_Check Loss On Init/results.json": [False, False],
    "art_checkpoints/MNISTModelNormalized_Check Loss On Init/results.json": [True],
    "art_checkpoints/MNISTModelNormalized_Overfit One Batch/results.json": [
        True,
        False,
        False,
    ],
    "art_checkpoints/MNISTModelNormalized_Overfit/results.json": [True],
    "art_checkpoints/MNISTModelNormalized_Regularize/results.json": [True],
}


def download_tutorial():
    """
    Downloads the tutorial from the GitHub repository using Cookiecutter.
    """
    try:
        cookiecutter(
            "https://github.com/SebChw/ART-Templates.git",
            no_input=True,
            extra_context={
                "project_name": "mnist_tutorial",
                "author": "test",
                "email": "test",
            },  # Pass the project_name to the template,
            checkout="mnist_tutorial_cookiecutter",  # Use the latest version of the template
        )
    except Exception as e:
        print("Error while generating project using Cookiecutter:", str(e))
        raise e


def clean_up():
    """
    Removes the downloaded tutorial directory.
    """
    print("Cleaning up...")
    shutil.rmtree("mnist_tutorial")


def run_jupyter_notebook():
    """
    Runs the Jupyter notebook for the downloaded tutorial.
    """
    os.chdir("mnist_tutorial")
    notebook_path = os.path.normpath("tutorial.ipynb")
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Create an ExecutePreprocessor instance
    executor = ExecutePreprocessor(timeout=800)
    try:
        executor.preprocess(notebook_content, {"metadata": {"path": "."}})
    except Exception as e:
        print("Error while running Jupyter notebook:", str(e))
        raise e
    os.chdir("..")


def check_outputs():
    """
    Checks the output files generated by the Jupyter notebook with expected output.
    """
    utils_folder = "tests/utils"
    jupyter_outputs_folder = "mnist_tutorial"
    print(os.getcwd())
    paths = open(os.path.join(utils_folder, "test_files.txt"), "r").read().splitlines()
    for path in paths:
        print(os.path.join(jupyter_outputs_folder, path))
        assert os.path.isfile(os.path.join(jupyter_outputs_folder, path))


def check_expected_steps_executions():
    jupyter_outputs_folder = "mnist_tutorial"
    for results_file, expected_results in EXPECTED_STEPS_EXECUTIONS.items():
        with open(os.path.join(jupyter_outputs_folder, results_file), "r") as f:
            results = json.load(f)

        for step_run, expected_result in zip(results["runs"], expected_results):
            assert step_run["successful"] == expected_result


def test_tutorial():
    """
    Tests the downloaded tutorial by running the Jupyter notebook and checking the output files.
    """
    if os.path.isdir("mnist_tutorial"):
        clean_up()
    print("Downloading tutorial...")
    download_tutorial()
    print("Running Jupyter notebook...")
    run_jupyter_notebook()
    print("Checking outputs...")
    check_outputs()
    check_expected_steps_executions()
    print("Cleaning up...")
    clean_up()


if __name__ == "__main__":
    test_tutorial()
