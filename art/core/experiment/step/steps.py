from abc import ABC, abstractmethod
from typing import Any, List, Dict

from lightning.pytorch import Trainer

from art.core.base_components.BaseModel import Baseline
from art.core.experiment.step.checks import Check
from art.core.experiment.step.step_savers import JSONStepSaver


class Step(ABC):
    name: str
    description: str

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def get_saved_state(self) -> Dict[str, str]:
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
        super().__init__("EvaluateBaselines", "Evaluates baselines on the dataset")
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
        JSONStepSaver().save(self.results, self.name, "results.json")

    def get_saved_state(self) -> Dict[str, str]:
        return {f"{baseline.name}_baseline": f"{baseline.name}_baseline/" for baseline in self.baselines} | {
            "results": "results.json",
        }


class CheckLossOnInit(Step):
    def __init__(self, model, datamodule):
        super().__init__("CheckLossOnInit", "Checks loss on init")
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()
        self.results = trainer.validate(
            model=self.model, dataloaders=self.datamodule.train_dataloader()
        )[0]
        JSONStepSaver().save(self.results, self.name, "results.json")

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }


class OverfitOneBatch(Step):
    def __init__(self, model, datamodule):
        super().__init__("OverfitOneBatch", "Overfits one batch")
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
        loss_at_the_end = float(trainer.logged_metrics["train_loss"])
        print(f"Loss at the end of overfitting: {loss_at_the_end}")
        JSONStepSaver().save({"loss_at_the_end":loss_at_the_end}, self.name, "results.json")

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }

class Overfit(Step):
    def __init__(self, model, datamodule, max_epochs=1):
        super().__init__("Overfit", "Overfits one batch")
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
        loss_at_the_end = float(self.results[0]["validation_loss"])
        print(f"Loss at the end of overfitting: {loss_at_the_end}")
        JSONStepSaver().save({"loss_at_the_end": loss_at_the_end}, self.name, "results.json")

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }


class Regularize(Step):
    def __init__(self, model, datamodule):
        super().__init__("Regularize", "Regularizes model")
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()  # It probably should take some configs
        trainer.fit(model=self.model, datamodule=self.datamodule)
        # TODO should we validate, or rather use trainer.logged_metrics["train_loss"]
        metrics = trainer.logged_metrics["validation_loss"]
        # TODO pass this loss somewhere to check if stage is passed succesfully.
        print(f"Loss at the end of regularization: {metrics}")
        JSONStepSaver().save({"metrics": metrics}, self.name, "results.json")

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }

class Tune(Step):
    def __init__(self, model, datamodule):
        super().__init__("Tune", "Tunes model")
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        trainer = Trainer()  # Here we should write other object for this.
        # TODO how to solve this?
        trainer.tune(model=self.model, datamodule=self.datamodule)

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }


class Squeeze(Step):
    pass
