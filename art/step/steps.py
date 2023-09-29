from typing import Dict

from lightning.pytorch import LightningDataModule, Trainer

from art.core.base_components.base_model import ArtModule
from art.utils.enums import TrainingStage
from art.step.step import Step


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""


class EvaluateBaseline(Step):
    """This class takes a baseline and evaluates/trains it on the dataset"""

    name = "Evaluate Baseline"
    description = "Evaluates a baseline on the dataset"

    def __init__(self, baseline: ArtModule, datamodule: LightningDataModule):
        trainer = Trainer(accelerator=baseline.device.type)
        super().__init__(baseline, datamodule, trainer)

    def do(self, previous_states: Dict):
        self.model.ml_train({"dataloader": self.datamodule.train_dataloader()})
        self.validate(trainer_kwargs={"datamodule": self.datamodule})


class CheckLossOnInit(Step):
    name = "Check Loss On Init"
    description = "Checks loss on init"

    def __init__(self, model: ArtModule, datamodule: LightningDataModule):
        super().__init__(model, datamodule, trainer=Trainer())

    def do(self, previous_states: Dict):
        train_loader = self.datamodule.train_dataloader()
        self.validate(trainer_kwargs={"dataloaders": train_loader})


class OverfitOneBatch(Step):
    name = "Overfit One Batch"
    description = "Overfits one batch"

    def __init__(
        self,
        model: ArtModule,
        datamodule: LightningDataModule,
        number_of_steps: int = 100,
    ):
        trainer = Trainer(overfit_batches=1, max_epochs=number_of_steps)
        super().__init__(model, datamodule, trainer)

    def do(self, previous_states: Dict):
        train_loader = self.datamodule.train_dataloader()
        self.train(trainer_kwargs={"train_dataloaders": train_loader})
        for key, value in self.trainer.logged_metrics.items():
            if hasattr(value, "item"):
                self.results[key] = value.item()
            else:
                self.results[key] = value


class Overfit(Step):
    name = "Overfit"
    description = "Overfits model"

    def __init__(
        self, model: ArtModule, datamodule: LightningDataModule, max_epochs: int = 1
    ):
        trainer = Trainer(max_epochs=max_epochs)
        super().__init__(model, datamodule, trainer)

    def validate_train(self, trainer_kwargs: Dict):
        self.current_stage = TrainingStage.TRAIN
        result = self.trainer.validate(model=self.model, **trainer_kwargs)
        self.results.update(result[0])

    def do(self, previous_states: Dict):
        train_loader = self.datamodule.train_dataloader()
        self.train(trainer_kwargs={"train_dataloaders": train_loader})
        self.validate_train(trainer_kwargs={"dataloaders": train_loader})
        self.validate(trainer_kwargs={"datamodule": self.datamodule})


class Regularize(Step):
    name = "Regularize"
    description = "Regularizes model"

    def __init__(
        self,
        model: ArtModule,
        datamodule: LightningDataModule,
        trainer_kwargs: Dict = {},
    ):
        trainer = Trainer(check_val_every_n_epoch=50, max_epochs=50, **trainer_kwargs)
        super().__init__(model, datamodule, trainer)
        self.model.turn_on_model_regularizations()
        self.datamodule.turn_on_regularizations()

    def do(self, previous_states: Dict):
        self.train(trainer_kwargs={"datamodule": self.datamodule})
        self.validate(trainer_kwargs={"datamodule": self.datamodule})


class Tune(Step):
    name = "Tune"
    description = "Tunes model"

    def __init__(self, model: ArtModule, datamodule: LightningDataModule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

    def do(self, previous_states: Dict):
        trainer = Trainer()  # Here we should write other object for this.
        # TODO how to solve this?
        trainer.tune(model=self.model, datamodule=self.datamodule)


class Squeeze(Step):
    pass
