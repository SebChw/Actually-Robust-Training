from typing import List

import numpy as np
import torch.nn as nn
from baselines import HeuristicBaseline, MlBaseline
from MyDataset import DummyDataModule
from MyModel import ClassificationModel
from sklearn.linear_model import LogisticRegression

from art.core.experiment.Experiment import Experiment
from art.core.experiment.step.steps import (
    CheckLossOnInit,
    EvaluateBaselines,
    Overfit,
    OverfitOneBatch,
    Regularize, Step,
)
from core.experiment.step.checks import Check, CheckScoreLessThan, CheckScoreExists


def calculate_metric(lightning_module, pred, gt):
    gt = gt.numpy()  # I know this can't be handled like that.
    metric1 = np.mean(pred == gt)
    lightning_module.log("metric1", metric1, on_step=False, on_epoch=True)
    # TODO steps should somehow share this.
    # Each experiment should have such function defined and all stages should reuse it


class MyExperiment(Experiment):
    def __init__(self, name, **kwargs):
        # Potentially we can save file versions to show which model etc was used.
        # TODO, do we want to use list or something different like self.add_step(). Consider builder pattern.
        self.steps = []
        self.checks = []
        # self.update_dashboard(self.steps) # now from each step we take internal information it has remembered and save them to show on a dashboard

    def add_step(self, step: Step, checks: List[Check]):
        self.steps.append(step)
        self.checks.append(checks)
    def run_all(self):
        for step, checks in zip(self.steps, self.checks):
            # Dependency injection so that user doesn't have to pass metric function everywhere
            step.set_metric(calculate_metric)
            step()
            #step.save_result() #TODO: should it be separate function, in my opinion not...
            print(step.name)
            for check in checks:
                result = check.check(None, step.get_saved_state())
                if not result.is_positive:
                    raise "eeeee"


        self.logger = None


data = DummyDataModule()

network = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 1))
model = ClassificationModel(
    model=network, loss_fn=nn.BCEWithLogitsLoss()
)

# TODO During overfitting and later stage we may need some additional kwargs to be passed to the trainer like max_epochs
# TODO Where this turn on regularization should be called? Probably it should be hidden from the user

# TODO: every step should do it. Experiment shuldn't know about dashboard existance

exp = MyExperiment("exp1")
exp.add_step(EvaluateBaselines([HeuristicBaseline(), MlBaseline(LogisticRegression())], DummyDataModule()), [])
exp.add_step(CheckLossOnInit(model, data), [CheckScoreExists("CheckLossOnInit", "", "validation_loss")])
exp.add_step(OverfitOneBatch(model, data), [CheckScoreLessThan("OverfitOneBatch", "desc", "loss_at_the_end", 0.01)])
exp.add_step(Overfit(model, data), [CheckScoreLessThan("Overfit", "desc", "loss_at_the_end", 0.3)])
exp.add_step(Regularize(model.turn_on_regularization(), data.turn_on_regularization()), [])
#exp.add_step(Tune(model, data), [])
exp.run_all()

