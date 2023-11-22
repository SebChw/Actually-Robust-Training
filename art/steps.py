import datetime
import gc
import hashlib
import inspect
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import lightning as L
from lightning import Trainer
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.loggers import Logger

from art.core import ArtModule
from art.loggers import (
    add_logger,
    art_logger,
    get_new_log_file_name,
    get_run_id,
    remove_logger,
)
from art.metrics import MetricCalculator
from art.utils.enums import TrainingStage
from art.utils.exceptions import MissingLogParamsException
from art.utils.paths import get_checkpoint_logs_folder_path
from art.utils.savers import JSONStepSaver


class NoModelUsed:
    pass


class Step(ABC):
    """
    An abstract base class representing a generic step in a project.
    """

    name = "Data analysis"
    model = NoModelUsed()

    def __init__(self):
        """
        Initialize the step with an empty results dictionary.
        """
        self.results = {
            "scores": {},
            "parameters": {},
            "timestamp": str(datetime.datetime.now()),
            "successful": False,
        }
        self.finalized = False
        self.model_name = ""

    def __call__(
        self,
        previous_states: Dict,
        datamodule: L.LightningDataModule,
        metric_calculator: MetricCalculator,
        run_id: Optional[str] = None,
    ):
        """
        Call the step and save its results.

        Args:
            previous_states (Dict): Dictionary containing the previous step states.
            datamodule (L.LightningDataModule): Data module to be used.
            metric_calculator (MetricCalculator): Metric calculator for this step.
        """
        log_file_name = get_new_log_file_name(
            run_id if run_id is not None else get_run_id()
        )
        logger_id = add_logger(
            get_checkpoint_logs_folder_path(self.get_full_step_name()) / log_file_name
        )
        try:
            self.datamodule = datamodule
            self.fill_basic_results()
            self.do(previous_states)
            self.finalized = True
        except Exception as e:
            art_logger.exception(f"Error while executing step {self.name}!")
            raise e
        finally:
            remove_logger(logger_id)
        self.results["log_file_name"] = log_file_name

    def fill_basic_results(self):
        """Fill basic results like hash and commit id"""
        self.results["hash"] = self.get_hash()
        try:
            self.results["commit_id"] = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except Exception:
            art_logger.exception("Error while getting commit id!")

    def get_full_step_name(self) -> str:
        """
        Retrieve the full name of the step, which is a combination of its ID and name.

        Returns:
            str: The full step name.
        """
        return self.name

    def get_hash(self) -> str:
        """
        Compute a hash based on the source code of the step's class.

        Returns:
            str: MD5 hash of the step's source code.
        """
        return hashlib.md5(
            inspect.getsource(self.__class__).encode("utf-8")
        ).hexdigest()

    def add_result(self, name: str, value: Any):
        """
        Add a result to the step's results dictionary.

        Args:
            name (str): Name of the result.
            value (Any): Value of the result.
        """
        self.results[name] = value

    def get_latest_run(self) -> Dict:
        """
        If step was run returns itself, otherwise returns the latest run from the JSONStepSaver.

        Returns:
            Dict: The latest run.
        """
        if self.finalized:
            return self.results
        return JSONStepSaver().load(self.get_full_step_name())["runs"][0]

    def was_run(self) -> bool:
        """
        Check if the step was already executed based on the existence of saved results.

        Returns:
            bool: True if the step was run, otherwise False.
        """
        path = JSONStepSaver().get_path(
            self.get_full_step_name(), JSONStepSaver.RESULT_NAME
        )
        return path.exists()

    def __repr__(self) -> str:
        """Representation of the step"""
        if not self.finalized:
            self.results["scores"] = self.get_latest_run()["scores"]
        result_repr = "\n".join(
            f"\t{k}: {v}" for k, v in self.results["scores"].items()
        )
        return f"Step: {self.name}, Model: {self.model_name}, Passed: {self.results['successful']}. Results:\n{result_repr}"

    def set_successful(self):
        self.results["successful"] = True

    def is_successful(self):
        return self.results["successful"]

    @abstractmethod
    def log_params(
        self,
    ):
        pass

    @abstractmethod
    def do(self, previous_states: Dict):
        """
        Abstract method to execute the step. Must be implemented by child classes.

        Args:
            previous_states (Dict): Dictionary containing the previous step states.
        """
        pass

    def save_to_disk(self):
        JSONStepSaver().save(self, self.get_full_step_name(), "results.json")


class ModelStep(Step):
    """
    A specialized step in the project, representing a model-based step.
    """

    def __init__(
        self,
        model_class: ArtModule,
        trainer_kwargs: Dict = {},
        model_kwargs: Dict = {},
        model_modifiers: List[Callable] = [],
        logger: Optional[Logger] = None,
    ):
        """
        Initialize a model-based step.

        Args:
            model_class (ArtModule): The model's class associated with this step.
            trainer_kwargs (Dict, optional): Arguments to be passed to the trainer. Defaults to {}.
            model_kwargs (Dict, optional): Arguments to be passed to the model. Defaults to {}.
            model_modifiers (List[Callable], optional): List of functions to be applied to the model. Defaults to [].
            datamodule_modifiers (List[Callable], optional): List of functions to be applied to the data module. Defaults to [].
            logger (Optional[Logger], optional): Logger to be used. Defaults to None.
        """
        super().__init__()
        if logger is not None:
            logger.add_tags(self.name)

        if not inspect.isclass(model_class):
            raise ValueError(
                "model_func must be class inhertiting from Art Module or path to the checkpoint. This is to avoid memory leaks. Simplest way of doing this is to use lambda function lambda : ArtModule()"
            )

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model_modifiers = model_modifiers
        self.logger = logger
        self.trainer_kwargs = trainer_kwargs

        self.model_name = model_class.__name__

    def __call__(
        self,
        previous_states: Dict,
        datamodule: L.LightningDataModule,
        metric_calculator: MetricCalculator,
        run_id: Optional[str] = None,
    ):
        """
        Call the model step, set the metric calculator for the model, and save the results.

        Args:
            previous_states (Dict): Dictionary containing the previous step states.
            datamodule (L.LightningDataModule): Data module to be used.
            metric_calculator (MetricCalculator): Metric calculator for this step.
        """
        self.trainer = Trainer(**self.trainer_kwargs, logger=self.logger)
        self.metric_calculator = metric_calculator
        curr_device = (
            "cuda" if isinstance(self.trainer.accelerator, CUDAAccelerator) else "cpu"
        )
        self.metric_calculator.to(curr_device)
        super().__call__(previous_states, datamodule, metric_calculator, run_id)
        del self.trainer
        gc.collect()

    def initialize_model(
        self,
    ) -> Optional[ArtModule]:
        """
        Initializes the model.
        """
        if self.trainer.model is not None:
            return None

        model = self.model_class(**self.model_kwargs)
        for modifier in self.model_modifiers:
            modifier(model)
        model.set_metric_calculator(self.metric_calculator)

        self.log_params(model)
        return model

    def train(self, trainer_kwargs: Dict):
        """
        Train the model using the provided trainer arguments.

        Args:
            trainer_kwargs (Dict): Arguments to be passed to the trainer for training the model.
        """
        self.trainer.fit(model=self.initialize_model(), **trainer_kwargs)
        logged_metrics = {k: v.item() for k, v in self.trainer.logged_metrics.items()}

        self.results["scores"].update(logged_metrics)
        self.results["model_path"] = self.trainer.checkpoint_callback.best_model_path

    def get_hash(self) -> str:
        """
        Compute a hash based on the source code of the step's class.

        Returns:
            str: MD5 hash of the step's source code.
        """
        return hashlib.md5(
            inspect.getsource(self.model_class).encode("utf-8")
        ).hexdigest()

    def validate(self, trainer_kwargs: Dict):
        """
        Validate the model using the provided trainer arguments.

        Args:
            trainer_kwargs (Dict): Arguments to be passed to the trainer for validating the model.
        """
        art_logger.info(f"Validating model {self.model_name}")

        result = self.trainer.validate(model=self.initialize_model(), **trainer_kwargs)
        self.results["scores"].update(result[0])

    def test(self, trainer_kwargs: Dict):
        """
        Test the model using the provided trainer arguments.

        Args:
            trainer_kwargs (Dict): Arguments to be passed to the trainer for testing the model.
        """
        result = self.trainer.test(model=self.initialize_model(), **trainer_kwargs)
        self.results["scores"].update(result[0])

    def get_full_step_name(self) -> str:
        """
        Retrieve the step ID, combining model name (if available) with the index.

        Returns:
            str: The step ID.
        """
        return f"{self.model_name}_{self.name}" if self.model_name != "" else self.name

    def get_current_stage(self) -> str:
        """
        Retrieve the current training stage of the trainer.

        Returns:
            str: Current training stage.
        """
        return self.trainer.state.stage.value

    def get_check_stage(self) -> str:
        """
        Get the validation stage value from the TrainingStage enum.

        Returns:
            str: Validation stage value.
        """
        return TrainingStage.VALIDATION.value

    def log_params(self, model):
        if hasattr(model, "log_params"):
            model_params = model.log_params()
            self.results["parameters"].update(model_params)

        else:
            raise MissingLogParamsException(
                "Model does not have log_params method. You don't want to regret lack of logs :)"
            )

        if hasattr(self.datamodule, "log_params"):
            data_params = self.datamodule.log_params()
            self.results["parameters"].update(data_params)
        else:
            raise MissingLogParamsException(
                "Datamodule does not have log_params method. You don't want to regret lack of logs :)"
            )

    def reset_trainer(self, logger: Optional[Logger] = None, trainer_kwargs: Dict = {}):
        """
        Reset the trainer.
        Args:
            trainer_kwargs (Dict): Arguments to be passed to the trainer.
            logger (Optional[Logger], optional): Logger to be used. Defaults to None.
        """
        self.trainer = Trainer(**trainer_kwargs, logger=logger)


class ExploreData(Step):
    """This class checks whether we have some markdown file description of the dataset + we implemented visualizations"""

    name = "Data analysis"
    description = "This step allows you to perform data analysis and extract information that is necessery in next steps"


class EvaluateBaseline(ModelStep):
    """This class takes a baseline and evaluates/trains it on the dataset"""

    name = "Evaluate Baseline"
    description = "Evaluates a baseline on the dataset"

    def __init__(
        self,
        baseline: ArtModule,
        device: Optional[str] = "cpu",
    ):
        super().__init__(baseline, {"accelerator": device})

    def do(self, previous_states: Dict):
        """
        This method evaluates baseline on the dataset

        Args:
            previous_states (Dict): previous states
        """
        art_logger.info("Training baseline")
        model = self.model_class()
        model.ml_train({"dataloader": self.datamodule.train_dataloader()})
        art_logger.info("Validating baseline")
        model.set_metric_calculator(self.metric_calculator)
        result = self.trainer.validate(model=model, datamodule=self.datamodule)
        self.results["scores"].update(result[0])


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
        number_of_steps: int = 50,
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

    def log_params(self, model):
        self.results["parameters"]["number_of_steps"] = self.number_of_steps
        super().log_params(model)


class Overfit(ModelStep):
    """This step tries to overfit the model"""

    name = "Overfit"
    description = "Overfits model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Logger] = None,
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

    def log_params(self, model):
        self.results["parameters"]["max_epochs"] = self.max_epochs
        super().log_params(model)


class Regularize(ModelStep):
    """This step tries applying regularization to the model"""

    name = "Regularize"
    description = "Regularizes model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Logger] = None,
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
        self.datamodule.turn_on_regularizations()
        art_logger.info("Training regularized model")
        self.train(trainer_kwargs={"datamodule": self.datamodule})

    def log_params(self, model):
        self.results["parameters"].update(self.trainer_kwargs)
        super().log_params(model)


class Tune(ModelStep):
    """This step tunes the model"""

    name = "Tune"
    description = "Tunes model"

    def __init__(
        self,
        model: ArtModule,
        logger: Optional[Logger] = None,
    ):
        super().__init__(model, logger=logger)

    def do(self, previous_states: Dict):
        """
        This method tunes the model

        Args:
            previous_states (Dict): previous states
        """
        # TODO how to solve this?


class Squeeze(ModelStep):
    pass


class TransferLearning(ModelStep):
    """This step tries performing proper transfer learning"""

    name = "TransferLearning"
    description = "This step tries performing proper transfer learning"

    def __init__(
        self,
        model: ArtModule,
        model_modifiers: List[Callable] = [],
        logger: Optional[Logger] = None,
        freezed_trainer_kwargs: Dict = {},
        unfreezed_trainer_kwargs: Dict = {},
        freeze_names: Optional[list[str]] = None,
        keep_unfrozen: Optional[int] = None,
        fine_tune_lr: float = 1e-5,
        fine_tune: bool = True,
    ):
        """
        This method initializes the step

        Args:
            model (ArtModule): model
            model_modifiers (List[Callable], optional): model modifiers. Defaults to [].
            logger (Logger, optional): logger. Defaults to None.
            freezed_trainer_kwargs (Dict, optional): trainer kwargs use for transfer learning with freezed weights. Defaults to {}.
            unfreezed_trainer_kwargs (Dict, optional): trainer kwargs use for fine tuning with unfreezed weights. Defaults to {}.
            freeze_names (Optional[list[str]], optional): name of model to freeze which appears in layers. Defaults to None.
            keep_unfrozen (Optional[int], optional): number of last layers to keep unfrozen. Defaults to None.
            fine_tune_lr (float, optional): fine tune lr. Defaults to 1e-5.
            fine_tune (bool, optional): whether or not perform fine tuning. Defaults to True.
        """
        super().__init__(
            model,
            trainer_kwargs=freezed_trainer_kwargs,
            logger=logger,
            model_modifiers=model_modifiers,
        )
        self.freeze_names = freeze_names
        self.keep_unfrozen = keep_unfrozen
        self.unfreezed_trainer_kwargs = unfreezed_trainer_kwargs
        self.fine_tune_lr = fine_tune_lr
        self.fine_tune = fine_tune

    def do(self, previous_states: Dict):
        """
        This method trains the model
        Args:
            previous_states (Dict): previous states
        """
        self.add_freezing()
        self.train(trainer_kwargs={"datamodule": self.datamodule})
        if self.fine_tune:
            self.add_unfreezing()
            self.reset_trainer(
                logger=self.trainer.logger, trainer_kwargs=self.unfreezed_trainer_kwargs
            )
            self.train(trainer_kwargs={"datamodule": self.datamodule})

    def log_params(self, model):
        self.results["parameters"].update(self.trainer_kwargs)
        super().log_params(model)

    def get_check_stage(self):
        """Returns check stage"""
        return TrainingStage.VALIDATION.value

    def add_freezing(self):
        """Adds freezing to the model"""

        def freeze_by_name(model):
            """Freeze parameters by layer names."""
            for name in self.freeze_names:
                for param in model.named_parameters():
                    if name in param[0]:
                        param[1].requires_grad = False

        def freeze_without_last_n(model):
            """Freeze all parameters except last n layers."""
            for i, param in enumerate(model.parameters()):
                if i < len(list(model.parameters())) - self.keep_unfrozen:
                    param.requires_grad = False

        def freeze_model(model):
            if self.freeze_names is not None and self.keep_unfrozen is not None:
                raise ValueError(
                    "Both freeze_names and keep_unfrozen are provided. Please provide only one of them."
                )
            elif self.freeze_names is not None:
                freeze_by_name(model)
            elif self.keep_unfrozen is not None:
                freeze_without_last_n(model)
            else:
                raise ValueError("No freezing criteria provided.")

        self.model_modifiers.append(freeze_model)

    def add_unfreezing(self):
        """Adds unfreezing to the model"""

        def unfreeze_model(model):
            for param in model.parameters():
                param.requires_grad = True

        self.model_modifiers.pop()
        self.model_modifiers.append(unfreeze_model)

    def add_lr_change(self):
        """Adds lr change to the model"""

        def change_lr(model):
            model.lr = self.fine_tune_lr

        self.model_modifiers.append(change_lr)
