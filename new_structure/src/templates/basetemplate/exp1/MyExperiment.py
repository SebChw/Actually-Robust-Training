import pandas as pd
from baselines import HeuristicBaseline, MlBaseline
from datasets import Dataset
from sklearn.datasets import make_classification
from src.core.experiment.Experiment import Experiment
from src.templates.basetemplate.exp1.MyModel import ClassificationModel
from src.templates.basetemplate.MyDataset import MyDataset

from new_structure.src.core.experiment.steps import (
    CheckLossOnInit,
    EvaluateBaselines,
    Overfit,
    OverfitOneBatch,
    Regularize,
    Tune,
)


def calculate_metrics(prediction, gt):
    # TODO steps should somehow share this.
    # Each experiment should have such function defined and all stages should reuse it
    pass


class MyExperiment(Experiment):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        X, y = make_classification(n_samples=10000)
        df = pd.DataFrame(X)
        df["y"] = y
        dataset = Dataset.from_pandas(df)

        self.network = None
        self.model = ClassificationModel(model=self.network)

        # Potentially we can save file versions to show which model etc was used.
        # TODO, do we want to use list or something different like self.add_step(). Consider builder pattern.
        self.steps = [
            EvaluateBaselines([HeuristicBaseline(), MlBaseline], dataset),
            CheckLossOnInit(
                self.model, dataset
            ),  # From now I assume that we will use the same mode
            OverfitOneBatch(self.model, dataset),
            Overfit(self.model, dataset),
            Regularize(
                self.model.turn_on_regularization(), dataset.turn_on_regularization()
            ),
            Tune(self.model, dataset),
        ]  # steps are run they remember their internal state

        # TODO: every step should do it. Experiment shuldn't know about dashboard existance
        # self.update_dashboard(self.steps) # now from each step we take internal information it has remembered and save them to show on a dashboard

        self.logger = None

        self.dataset = MyDataset()
