from abc import ABC, abstractmethod
from typing import Any, Dict, List

from lightning.pytorch import Trainer

from art.enums import TRAIN_LOSS, VALIDATION_LOSS, TrainingStage
from art.experiment_state import ExperimentState
from art.step.checks import Check
from art.step.step_savers import JSONStepSaver


class Step(ABC):
    name: str
    description: str
    STEPS_REGISTRY = []

    def __init__(self):
        self.STEPS_REGISTRY.append(self)

    @abstractmethod
    def __call__(self):
        ExperimentState.current_stage = TrainingStage.VALIDATION
        ExperimentState.current_step = self

    @abstractmethod
    def get_saved_state(self) -> Dict[str, str]:
        pass


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""


# TODO move init to Step class, as all steps have same init
# TODO Add saving results


class EvaluateBaselines(Step):
    """This class takes list of baselines and evaluates/trains them on the dataset"""

    name = "Evaluate Baselines"
    description = "Evaluates baselines on the dataset"

    def __init__(
        self, baselines: List, datamodule
    ):  # Probably all steps could have same init
        super().__init__()
        self.baselines = baselines
        self.datamodule = datamodule

    def __call__(
        self,
    ):  # Probably all steps could have same loop and saving results etc.
        super().__call__()
        self.results = {}
        for baseline in self.baselines:
            baseline.ml_train({"dataloader": self.datamodule.train_dataloader()})

            trainer = Trainer(accelerator=baseline.device.type)
            results = trainer.validate(model=baseline, datamodule=self.datamodule)

            # TODO: how to save results in a best way?
            # TODO do it on the fly in some files. After every step some results are saved in a file
            self.results[baseline.name] = results
        JSONStepSaver().save(self.results, self.name, "results.json")

    def get_saved_state(self) -> Dict[str, str]:
        return {
            f"{baseline.name}_baseline": f"{baseline.name}_baseline/"
            for baseline in self.baselines
        } | {
            "results": "results.json",
        }


class CheckLossOnInit(Step):
    name = "Check Loss On Init"
    description = "Checks loss on init"

    def __init__(self, model, datamodule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        super().__call__()
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
    name = "Overfit One Batch"
    description = "Overfits one batch"

    def __init__(self, model, datamodule, number_of_steps=100):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.number_of_steps = number_of_steps

    def __call__(self):
        super().__call__()
        trainer = Trainer(overfit_batches=1, max_epochs=self.number_of_steps)
        trainer.fit(
            model=self.model, train_dataloaders=self.datamodule.train_dataloader()
        )

        # this contains loss after last step. It should be very small
        # additionally the name of the metric should be predefined
        # TODO change "train_loss" to some constant
        loss_at_the_end = float(trainer.logged_metrics[TRAIN_LOSS])
        print(f"Loss at the end of overfitting: {loss_at_the_end}")
        JSONStepSaver().save(
            {"loss_at_the_end": loss_at_the_end}, self.name, "results.json"
        )

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }


class Overfit(Step):
    name = "Overfit"
    description = "Overfits model"

    def __init__(self, model, datamodule, max_epochs=1):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs

    def __call__(self):
        super().__call__()
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
        loss_at_the_end = float(self.results[0][VALIDATION_LOSS])
        print(f"Loss at the end of overfitting: {loss_at_the_end}")
        JSONStepSaver().save(
            {"loss_at_the_end": loss_at_the_end}, self.name, "results.json"
        )

    def get_saved_state(self) -> Dict[str, str]:
        return {
            "results": "results.json",
        }


class Regularize(Step):
    name = "Regularize"
    description = "Regularizes model"

    def __init__(self, model, datamodule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

    def __call__(self):
        super().__call__()
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
    name = "Tune"
    description = "Tunes model"

    def __init__(self, model, datamodule):
        super().__init__()
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
