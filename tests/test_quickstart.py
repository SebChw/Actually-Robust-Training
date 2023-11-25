import math
import shutil

import pytest
import torch.nn as nn
from torchmetrics import Accuracy
from utils import check_expected_steps_executions, clean_up

from art.checks import CheckScoreCloseTo, CheckScoreGreaterThan, CheckScoreLessThan
from art.metrics import SkippedMetric
from art.project import ArtProject
from art.steps import CheckLossOnInit, Overfit, OverfitOneBatch
from art.utils.quickstart import ArtModuleExample, LightningDataModuleExample

EXPECTED_STEPS_EXECUTIONS = {
    "art_checkpoints/ArtModuleExample_Check Loss On Init/results.json": [True],
    "art_checkpoints/ArtModuleExample_Overfit One Batch/results.json": [True],
    "art_checkpoints/ArtModuleExample_Overfit/results.json": [False],
}


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    clean_up("art_checkpoints")
    clean_up("lightning_logs")


def test_quickstart():
    # Initialize the datamodule, and indicate the model class
    datamodule = LightningDataModuleExample()
    model_class = ArtModuleExample

    # Define metrics and loss functions to be calculated within the project
    metric = Accuracy(task="multiclass", num_classes=datamodule.n_classes)
    loss_fn = nn.CrossEntropyLoss()

    # Create an ART project and register these metrics
    project = ArtProject(name="quickstart", datamodule=datamodule)
    project.register_metrics([metric, loss_fn])

    # Add steps to the project
    EXPECTED_LOSS = -math.log(1 / datamodule.n_classes)
    project.add_step(
        CheckLossOnInit(model_class),
        checks=[CheckScoreCloseTo(loss_fn, EXPECTED_LOSS, rel_tol=0.01)],
        skipped_metrics=[SkippedMetric(metric)],
    )
    project.add_step(
        OverfitOneBatch(model_class, number_of_steps=100),
        checks=[CheckScoreLessThan(loss_fn, 0.1)],
        skipped_metrics=[SkippedMetric(metric)],
    )
    project.add_step(
        Overfit(model_class, max_epochs=10),
        checks=[CheckScoreGreaterThan(metric, 0.9)],
    )

    # Run your experiment
    project.run_all()

    # Check failed for step: Overfit. Reason: Score 0.7900000214576721 is not greater than 0.9
    # Summary:
    # Step: Check Loss On Init, Model: ArtModuleExample, Passed: True. Results:
    #         CrossEntropyLoss-validate: 2.299098491668701
    # Step: Overfit One Batch, Model: ArtModuleExample, Passed: True. Results:
    #         CrossEntropyLoss-train: 0.03459629788994789
    # Step: Overfit, Model: ArtModuleExample, Passed: False. Results:
    #         MulticlassAccuracy-train: 0.7900000214576721
    #         CrossEntropyLoss-train: 0.5287203788757324
    #         MulticlassAccuracy-validate: 0.699999988079071
    #         CrossEntropyLoss-validate: 0.8762148022651672

    check_expected_steps_executions("", EXPECTED_STEPS_EXECUTIONS)
