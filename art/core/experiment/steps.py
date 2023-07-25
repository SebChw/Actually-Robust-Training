from abc import ABC, abstractmethod
from typing import Any, List

from lightning.pytorch import Trainer

from art.new_structure.core.base_components.BaseModel import Baseline
from art.new_structure.core.experiment.Check import Check


class Step(ABC):
    name: str
    descrption: str
    checks: Check

    # @abstractmethod
    # def verify_passed(self) -> bool:
    #     pass

    @abstractmethod
    def __call__(self):
        pass

    # TODO read about properties etc. and select the best
    def set_metric(self, metric):
        self.metric = metric


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""


# TODO move init to Step class, as all steps have same init
# TODO Add saving results


class EvaluateBaselines(Step):
    """This class takes list of baselines and evaluates/trains them on the dataset"""

    def __init__(
        self, baselines: List[Baseline], datamodule
    ):  # Probably all steps could have same init
        self.baselines = baselines
        # TODO should we enforce datamodule to be hf dataset?
        self.datamodule = datamodule

    def __call__(
        self,
    ):  # Probably all steps could have same loop and saving results etc.
        self.results = {}
        for baseline in self.baselines:
            baseline.train_baseline(self.datamodule.train_dataloader())
            baseline.set_metric(self.metric)

            trainer = Trainer(accelerator=baseline.accelerator)
            results = trainer.validate(model=baseline, datamodule=self.datamodule)

            # TODO: how to save results in a best way?
            # TODO do it on the fly in some files. After every step some results are saved in a file
            self.results[baseline.name] = results


class CheckLossOnInit(Step):
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()
        self.results = trainer.validate(
            model=self.model, dataloaders=self.datamodule.train_dataloader()
        )


class OverfitOneBatch(Step):
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer(overfit_batches=1, max_epochs=50)
        trainer.fit(
            model=self.model, train_dataloaders=self.datamodule.train_dataloader()
        )


class Overfit(Step):
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()  # It probably should take some configs
        trainer.fit(
            model=self.model, train_dataloaders=self.datamodule.get_subset("train")
        )
        self.results = trainer.validate(
            model=self.model, datamodule=self.datamodule.get_subset("train")
        )


class Regularize(Step):
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()  # It probably should take some configs
        trainer.fit(model=self.model, datamodule=self.datamodule)


class Tune(Step):
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()  # Here we should write other object for this.
        # TODO how to solve this?
        trainer.tune(model=self.model, datamodule=self.datamodule)


class Squeeze(Step):
    pass
