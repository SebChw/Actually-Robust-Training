from src.core.experiment.Experiment import Experiment
from src.core.experiment.Step import Step
from src.templates.basetemplate.MyDataset import MyDataset
from src.templates.basetemplate.exp1.MyModel import MyModel


class MyExperiment(Experiment):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.steps = [
            Step(),
            Step(),
            Step(),
            Step()
        ]
        self.logger = None
        self.model = MyModel()
        self.dataset = MyDataset()

