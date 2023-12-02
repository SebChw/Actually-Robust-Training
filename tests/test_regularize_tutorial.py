import os

import pytest
from utils import check_expected_steps_executions, clean_up, download_tutorial

# We run just single epoch that is why everything fails.
EXPECTED_STEPS_EXECUTIONS = {
    "art_checkpoints/FoodClassifier_Regularize/results.json": [
        False,
        False,
        False,
        False,
        False,
    ],
}

TUT_NAME = "regularize_tutorial"


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    clean_up(TUT_NAME)


def test_tutorial():
    download_tutorial(proj_name=TUT_NAME, branch=TUT_NAME)
    os.chdir(TUT_NAME)
    os.system("pip install kornia")
    os.system("python run.py --max_epochs 1")
    os.chdir("..")
    check_expected_steps_executions(TUT_NAME, EXPECTED_STEPS_EXECUTIONS)
