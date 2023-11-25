import json
import os
import shutil


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
