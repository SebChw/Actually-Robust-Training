from typing import Any, Dict, Iterable, Optional, Union

from lightning.pytorch.loggers import Logger

from art.art_logger import art_logger
from art.core.base_components.base_model import ArtModule
from art.step.step import ModelStep, Step
from art.utils.enums import TrainingStage


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""

    name = "Data analysis"
    description = "This step allows you to perform data analysis and extract information that is necessery in next steps"

    def get_step_id(self) -> str:
        """
        Returns step id

        Returns:
            str: step id
        """
        return f"data_analysis"


class EvaluateBaseline(ModelStep):
    """This class takes a baseline and evaluates/trains it on the dataset"""

    name = "Evaluate Baseline"
    description = "Evaluates a baseline on the dataset"

    def __init__(
        self,
        baseline: ArtModule,
    ):
        super().__init__(baseline, {"accelerator": baseline.device.type})

    def do(self, previous_states: Dict):
        """
        This method evaluates baseline on the dataset

        Args:
            previous_states (Dict): previous states
        """
        art_logger.info("Training baseline")
        self.model.ml_train({"dataloader": self.datamodule.train_dataloader()})
        art_logger.info("Validating baseline")
        self.validate(trainer_kwargs={"datamodule": self.datamodule})


class CheckLossOnInit(ModelStep):
    """This step checks whether the loss on init is as expected"""

    name = "Check Loss On Init"
    description = "Checks loss on init"

    def __init__(
        self,
        model: ArtModule,
    ):
        super().__init__(model)

    def do(self, previous_states: Dict):
        """
        This method checks loss on init. It validates the model on the train dataloader and checks whether the loss is as expected.

        Args:
            previous_states (Dict): previous states
        """
        train_loader = self.datamodule.train_dataloader()
        art_logger.info("Calculating loss on init")
        self.validate(trainer_kwargs={"dataloaders": train_loader})


class OverfitOneBatch(ModelStep):
    """This step tries to Overfit one train batch"""

    name = "Overfit One Batch"
    description = "Overfits one batch"

    def __init__(
        self,
        model: ArtModule,
        number_of_steps: int = 100,
    ):
        self.number_of_steps = number_of_steps
        super().__init__(model, {"overfit_batches": 1, "max_epochs": number_of_steps})

    def do(self, previous_states: Dict):
        """
        This method overfits one batch

        Args:
            previous_states (Dict): previous states
        """
        train_loader = self.datamodule.train_dataloader()
        art_logger.info("Overfitting one batch")
        self.train(trainer_kwargs={"train_dataloaders": train_loader})
        for key, value in self.trainer.logged_metrics.items():
            if hasattr(value, "item"):
                self.results[key] = value.item()
            else:
                self.results[key] = value

    def get_check_stage(self):
        """Returns check stage"""
        return TrainingStage.TRAIN.value

    def log_params(self):
        self.results["parameters"]["number_of_steps"] = self.number_of_steps
        super().log_params()


class Overfit(ModelStep):
    """This step tries to overfit the model"""

    name = "Overfit"
    description = "Overfits model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        max_epochs: int = 1,
    ):
        self.max_epochs = max_epochs

        super().__init__(model, {"max_epochs": max_epochs}, logger=logger)

    def do(self, previous_states: Dict):
        """
        This method overfits the model

        Args:
            previous_states (Dict): previous states
        """
        train_loader = self.datamodule.train_dataloader()
        art_logger.info("Overfitting model")
        self.train(trainer_kwargs={"train_dataloaders": train_loader})
        art_logger.info("Validating overfitted model")
        self.validate(trainer_kwargs={"datamodule": self.datamodule})

    def get_check_stage(self):
        """Returns check stage"""
        return TrainingStage.TRAIN.value

    def log_params(self):
        self.results["parameters"]["max_epochs"] = self.max_epochs
        super().log_params()


class Regularize(ModelStep):
    """This step tries applying regularization to the model"""

    name = "Regularize"
    description = "Regularizes model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        trainer_kwargs: Dict = {},
    ):
        self.trainer_kwargs = trainer_kwargs
        super().__init__(model, trainer_kwargs, logger=logger)

    def do(self, previous_states: Dict):
        """
        This method regularizes the model

        Args:
            previous_states (Dict): previous states
        """
        art_logger.info("Turning on regularization")
        self.model.turn_on_model_regularizations()
        self.datamodule.turn_on_regularizations()
        art_logger.info("Training regularized model")
        self.train(trainer_kwargs={"datamodule": self.datamodule})

    def log_params(self):
        self.results["parameters"].update(self.trainer_kwargs)
        super().log_params()


class Tune(ModelStep):
    """This step tunes the model"""

    name = "Tune"
    description = "Tunes model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
    ):
        super().__init__(model=model, logger=logger)

    def do(self, previous_states: Dict):
        """
        This method tunes the model

        Args:
            previous_states (Dict): previous states
        """
        # TODO how to solve this?


class Squeeze(ModelStep):
    pass
