from typing import Dict, Iterable, Optional, Union

from lightning import Trainer
from lightning.pytorch.loggers import Logger

from art.core.base_components.base_model import ArtModule
from art.step.step import Step
from art.utils.enums import TrainingStage


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""


class EvaluateBaseline(Step):
    """This class takes a baseline and evaluates/trains it on the dataset"""

    name = "Evaluate Baseline"
    description = "Evaluates a baseline on the dataset"

    def __init__(
        self,
        baseline: ArtModule,
    ):
        trainer = Trainer(accelerator=baseline.device.type)
        super().__init__(baseline, trainer)

    def do(self, previous_states: Dict):
        self.model.ml_train({"dataloader": self.datamodule.train_dataloader()})
        self.validate(trainer_kwargs={"datamodule": self.datamodule})


class CheckLossOnInit(Step):
    name = "Check Loss On Init"
    description = "Checks loss on init"

    def __init__(
        self,
        model: ArtModule,
    ):
        super().__init__(model, trainer=Trainer())

    def do(self, previous_states: Dict):
        train_loader = self.datamodule.train_dataloader()
        self.validate(trainer_kwargs={"dataloaders": train_loader})


class OverfitOneBatch(Step):
    name = "Overfit One Batch"
    description = "Overfits one batch"

    def __init__(
        self,
        model: ArtModule,
        number_of_steps: int = 100,
    ):
        trainer = Trainer(overfit_batches=1, max_epochs=number_of_steps)
        super().__init__(model, trainer)

    def do(self, previous_states: Dict):
        train_loader = self.datamodule.train_dataloader()
        self.train(trainer_kwargs={"train_dataloaders": train_loader})
        for key, value in self.trainer.logged_metrics.items():
            if hasattr(value, "item"):
                self.results[key] = value.item()
            else:
                self.results[key] = value

    def get_check_stage(self):
        return TrainingStage.TRAIN.value


class Overfit(Step):
    name = "Overfit"
    description = "Overfits model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        max_epochs: int = 1,
    ):
        trainer = Trainer(max_epochs=max_epochs, logger=logger)
        super().__init__(model, trainer)

    def do(self, previous_states: Dict):
        train_loader = self.datamodule.train_dataloader()
        self.train(trainer_kwargs={"train_dataloaders": train_loader})
        self.validate(trainer_kwargs={"datamodule": self.datamodule})

    def get_check_stage(self):
        return TrainingStage.TRAIN.value


class Regularize(Step):
    name = "Regularize"
    description = "Regularizes model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        trainer_kwargs: Dict = {},
    ):
        trainer = Trainer(**trainer_kwargs, logger=logger)
        super().__init__(model, trainer)

    def do(self, previous_states: Dict):
        self.model.turn_on_model_regularizations()
        self.datamodule.turn_on_regularizations()
        self.train(trainer_kwargs={"datamodule": self.datamodule})


class Tune(Step):
    name = "Tune"
    description = "Tunes model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
    ):
        super().__init__(model=model, trainer=Trainer(logger=logger))
        self.logger = logger

    def do(self, previous_states: Dict):
        trainer = Trainer(
            logger=self.logger
        )  # Here we should write other object for this.
        # TODO how to solve this?
        trainer.tune(model=self.model, datamodule=self.datamodule)


class Squeeze(Step):
    pass
