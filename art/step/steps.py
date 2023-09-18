from abc import ABC, abstractmethod
from typing import Any, Dict, List

from lightning.pytorch import Trainer

from art.enums import TRAIN_LOSS, VALIDATION_LOSS, TrainingStage
from art.step.checks import Check
from art.step.step import Step


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""


# TODO add something like Trainer kwargs to every step


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

    def do(
        self, previous_states
    ):  # Probably all steps could have same loop and saving results etc.
        self.results = {}
        for baseline in self.baselines:
            baseline.ml_train({"dataloader": self.datamodule.train_dataloader()})

            trainer = Trainer(accelerator=baseline.device.type)
            results = trainer.validate(model=baseline, datamodule=self.datamodule)

            # TODO: how to save results in a best way?
            # TODO do it on the fly in some files. After every step some results are saved in a file
            print(results)
            self.results[baseline.name] = results

    def get_saved_state(self) -> Dict[str, str]:
        return {
            f"{baseline.name}_baseline": f"{baseline.name}_baseline/"
            for baseline in self.baselines
        }

    def get_step_id(self) -> str:
        baseline_prefix = "_".join(
            [baseline.__class__.__name__ for baseline in self.baselines]
        )
        datamodule_prefix = self.datamodule.__class__.__name__
        return f"{baseline_prefix}_{datamodule_prefix}"


class CheckLossOnInit(Step):
    name = "Check Loss On Init"
    description = "Checks loss on init"

    def __init__(self, model, datamodule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

    def do(self, previous_states):
        trainer = Trainer()
        self.results.update(
            trainer.validate(
                model=self.model, dataloaders=self.datamodule.train_dataloader()
            )[0]
        )


class OverfitOneBatch(Step):
    name = "Overfit One Batch"
    description = "Overfits one batch"

    def __init__(self, model, datamodule, number_of_steps=100):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.number_of_steps = number_of_steps

    def do(self, previous_states):
        trainer = Trainer(overfit_batches=1, max_epochs=self.number_of_steps)
        trainer.fit(
            model=self.model, train_dataloaders=self.datamodule.train_dataloader()
        )
        self.results.update(trainer.logged_metrics)


class Overfit(Step):
    name = "Overfit"
    description = "Overfits model"

    def __init__(self, model, datamodule, max_epochs=1):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs

    def do(self, previous_states):
        # It probably should take some configs
        trainer = Trainer(max_epochs=self.max_epochs)
        trainer.fit(
            model=self.model, train_dataloaders=self.datamodule.train_dataloader()
        )

        self.results = trainer.validate(
            model=self.model, dataloaders=self.datamodule.train_dataloader()
        )[0]
        self.experiment.current_stage = TrainingStage.VALIDATION
        self.results.update(
            trainer.validate(
                model=self.model, dataloaders=self.datamodule.val_dataloader()
            )[0]
        )


class Regularize(Step):
    name = "Regularize"
    description = "Regularizes model"

    def __init__(self, model, datamodule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

    def do(self, previous_states):
        self.model.turn_on_model_regularizations()
        self.datamodule.turn_on_regularizations()

        trainer = Trainer(
            check_val_every_n_epoch=50, max_epochs=50
        )  # TODO It probably should take some configs
        trainer.fit(model=self.model, datamodule=self.datamodule)
        self.experiment.current_stage = TrainingStage.VALIDATION
        self.results = trainer.validate(
            model=self.model, dataloaders=self.datamodule.val_dataloader()
        )[0]


class Tune(Step):
    name = "Tune"
    description = "Tunes model"

    def __init__(self, model, datamodule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

    def do(self, previous_states):
        trainer = Trainer()  # Here we should write other object for this.
        # TODO how to solve this?
        trainer.tune(model=self.model, datamodule=self.datamodule)


class Squeeze(Step):
    pass
