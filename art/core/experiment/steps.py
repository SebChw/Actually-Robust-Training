from abc import ABC, abstractmethod
from typing import Any, List

from lightning.pytorch import Trainer

from art.core.base_components.BaseModel import Baseline
from art.core.experiment.Check import Check


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

        # this contains loss after last step. It should be very small
        # additionally the name of the metric should be predefined
        # TODO think if we want to measure some metrics here to.
        loss_at_the_end = trainer.logged_metrics["train_loss"]
        print(f"Loss at the end of overfitting: {loss_at_the_end}")


class Overfit(Step):
    def __init__(self, model, datamodule, max_epochs=1):
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs

    def __call__(self):
        # It probably should take some configs
        trainer = Trainer(max_epochs=self.max_epochs)
        trainer.fit(
            model=self.model, train_dataloaders=self.datamodule.train_dataloader()
        )
        # TODO should we validate, or rather use trainer.logged_metrics["train_loss"]
        self.results = trainer.validate(
            model=self.model, dataloaders=self.datamodule.train_dataloader()
        )
        # TODO pass this loss somewhere to check if stage is passed succesfully.
        loss_at_the_end = self.results[0]["validation_loss"]
        print(f"Loss at the end of overfitting: {loss_at_the_end}")


class Regularize(Step):
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()  # It probably should take some configs
        trainer.fit(model=self.model, datamodule=self.datamodule)
        # TODO should we validate, or rather use trainer.logged_metrics["train_loss"]
        metrics = trainer.logged_metrics["validation_loss"]
        # TODO pass this loss somewhere to check if stage is passed succesfully.
        print(f"Loss at the end of regularization: {metrics}")


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
