import numpy as np
import torch.nn as nn
from baselines import HeuristicBaseline, MlBaseline
from MyDataset import DummyDataModule
from MyModel import ClassificationModel
from sklearn.linear_model import LogisticRegression

from art.new_structure.core.experiment.Experiment import Experiment
from art.new_structure.core.experiment.steps import (
    CheckLossOnInit,
    EvaluateBaselines,
    Overfit,
    OverfitOneBatch,
    Regularize,
    Tune,
)


def calculate_metric(lightning_module, pred, gt):
    gt = gt.numpy()  # I know this can't be handled like that.
    metric1 = np.mean(pred == gt)
    lightning_module.log("metric1", metric1, on_step=False, on_epoch=True)
    # TODO steps should somehow share this.
    # Each experiment should have such function defined and all stages should reuse it


class MyExperiment(Experiment):
    def __init__(self, name, **kwargs):
        data = DummyDataModule()

        self.network = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 1))
        self.model = ClassificationModel(
            model=self.network, loss_fn=nn.BCEWithLogitsLoss()
        )

        # Potentially we can save file versions to show which model etc was used.
        # TODO, do we want to use list or something different like self.add_step(). Consider builder pattern.
        self.steps = [
            EvaluateBaselines(
                [HeuristicBaseline(), MlBaseline(LogisticRegression())], data
            ),
            CheckLossOnInit(
                self.model, data
            ),  # From now I assume that we will use the same mode
            OverfitOneBatch(self.model, data),
            # Overfit(self.model, data),
            # Regularize(
            #    self.model.turn_on_regularization(), data.turn_on_regularization()
            # ),
            # Tune(self.model, data),
        ]  # steps are run they remember their internal state

        # TODO: every step should do it. Experiment shuldn't know about dashboard existance
        # self.update_dashboard(self.steps) # now from each step we take internal information it has remembered and save them to show on a dashboard

        for step in self.steps:
            # Dependency injection so that user doesn't have to pass metric function everywhere
            step.set_metric(calculate_metric)
            step()

        self.logger = None


exp = MyExperiment("exp1")
