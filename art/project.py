from collections import defaultdict
from typing import Any, Dict, List, Union

import lightning as L

from art.checks import Check
from art.loggers import (
    add_logger,
    art_logger,
    get_new_log_file_name,
    get_run_id,
    remove_logger,
)
from art.metrics import MetricCalculator, SkippedMetric
from art.steps import ModelStep, Step
from art.utils.enums import TrainingStage
from art.utils.exceptions import CheckFailedException
from art.utils.paths import EXPERIMENT_LOG_DIR


class ArtProjectState:
    current_step: Union[Step, None]
    current_stage: TrainingStage = TrainingStage.TRAIN
    step_states: Dict[str, Dict[str, Dict[str, str]]]
    status: str
    """
    A class for managing the state of a project.
        steps:{
        "model_name": {
            "step_name": {/*step state*/},
            "step_name2": {/*step state*/},
        }
        "model2_name: {
            "step_name": {/*step state*/},
            "step_name2": {/*step state*/},
        }
    }"""

    def __init__(self):
        self.step_states = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        self.status = "created"
        self.current_step = None

    def get_steps(self):
        """
        Returns all steps that were run

        Returns:
            Dict[str, Dict[str, Dict[str, str]]]: [description]
        """
        return self.step_states

    def add_step(self, step):
        """
        Adds step to the state

        Args:
            step (Step): A step to be add the the project
        """
        self.step_states.append(step)

    def get_current_step(self):
        """
        Gets current step

        Returns:
            Step: Current step
        """
        return self.current_step.name

    def get_current_stage(self):
        """
        Gets current stage

        Returns:
            TrainingStage: Current stage
        """
        return self.current_step.get_current_stage()


class ArtProject:
    """
    Represents a single Art project, encapsulating steps, state, metrics, and logging.
    """

    def __init__(self, name: str, datamodule: L.LightningDataModule, **kwargs):
        """
        Initialize an Art project.

        Args:
            name (str): The name of the project.
            datamodule (L.LightningDataModule): Data module to be used in this project.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.steps: List[Dict] = []
        self.datamodule = datamodule
        self.state = ArtProjectState()
        self.metric_calculator = MetricCalculator(self)
        self.changed_steps: List[str] = []

    def add_step(
        self,
        step: "Step",
        checks: List[Check],
        skipped_metrics: List[SkippedMetric] = [],
    ):
        """
        Add a step to the project.

        Args:
            step (Step): The step to be added.
            checks (List[Check]): A list of checks associated with the step.
            skipped_metrics (List[SkippedMetric]): A list of metrics to skip for this step.
        """
        self.steps.append(
            {
                "step": step,
                "checks": checks,
                "skipped_metrics": skipped_metrics,
            }
        )

    def fill_step_states(self, step: "Step"):
        """
        Update step states with the results from the given step.

        Args:
            step (Step): The step whose results need to be recorded.
        """
        self.state.step_states[step.model_name][
            step.get_full_step_name()
        ] = step.get_latest_run()

    def check_checks(self, step: "Step", checks: List[Check]):
        """
        Validate if all checks pass for a given step.

        Args:
            step (Step): The step to check.
            checks (List[Check]): List of checks to validate.

        Raises:
            Exception: If any of the checks fail.
        """
        for check in checks:
            result = check.check(step)
            if not result.is_positive:
                msg = f"Check failed for step: {step.name}. Reason: {result.error}"
                raise CheckFailedException(msg)
        step.set_successful()

    def check_if_must_be_run(self, step: "Step", checks: List[Check]) -> bool:
        """
        Check if a given step needs to be executed or if it can be skipped.

        Args:
            step (Step): The step to check.
            checks (List[Check]): List of checks to validate.

        Returns:
            bool: True if the step must be run, False otherwise.
        """
        if not step.was_run():
            return True
        else:
            step_current_hash = step.get_hash()
            step_saved_hash = step.get_latest_run()["hash"]
            model_changed = True if step_current_hash != step_saved_hash else False
            if model_changed:
                self.changed_steps.append(step.get_full_step_name())
            try:
                self.check_checks(step, checks)
            except CheckFailedException:
                return True

            return False

    def run_all(self, force_rerun=False):
        """
        Execute all steps in the project.

        Args:
            force_rerun (bool): Whether to force rerun all steps.
        """
        run_id = get_run_id()
        logger_id = add_logger(EXPERIMENT_LOG_DIR / get_new_log_file_name(run_id))
        self.changed_steps = []
        try:
            for step_dict in self.steps:
                self.metric_calculator.compile(step_dict["skipped_metrics"])
                step, checks = step_dict["step"], step_dict["checks"]
                self.state.current_step = step

                rerun_step = self.check_if_must_be_run(step, checks)

                if not rerun_step and not force_rerun:
                    self.fill_step_states(step)
                    continue
                try:
                    step(
                        self.state.step_states,
                        self.datamodule,
                        self.metric_calculator,
                        run_id,
                    )
                    self.check_checks(step, checks)
                except CheckFailedException as e:
                    art_logger.warning(e)
                    step.save_to_disk()
                    break

                self.fill_step_states(step)
                step.save_to_disk()

            self.print_summary()
        except Exception as e:
            raise e
        finally:
            remove_logger(logger_id)

    def print_summary(self):
        """
        Prints a summary of the project.
        """
        art_logger.info("Summary: ")
        for step in self.steps:
            art_logger.info(step["step"])
            if not step["step"].is_successful():
                break
        if len(self.changed_steps) > 0:
            art_logger.info(
                f"Code of the following steps was changed: {', '.join(self.changed_steps)}\n Rerun could be needed."
            )

    def get_steps(self):
        """
        Retrieve all steps in the project.

        Returns:
            List[Dict[str, Any]]: List of steps.
        """
        return self.steps

    def get_step(self, step_id: int) -> "Step":
        """
        Retrieve a specific step by its ID.

        Args:
            step_id (int): The ID of the step to retrieve.

        Returns:
            Step: The specified step.
        """
        return self.steps[step_id]["step"]

    def replace_step(self, step: "Step", step_id: int = -1):
        """
        Replace an existing step with a new one.

        Args:
            step (Step): The new step.
            step_id (int): The ID of the step to replace. Default is the last step.
        """
        self.steps[step_id]["step"] = step

    def register_metrics(self, metrics: List[Any]):
        """
        Register metrics to the project.

        Args:
            metrics (List[Any]): A list of metrics to be registered.
        """
        self.metric_calculator.add_metrics(metrics)

    def to(self, device: str):
        """
        Move the metric calculator to a specified device.

        Args:
            device (str): The device to move the metrics to.
        """
        self.metric_calculator.to(device)

    def update_datamodule(self, datamodule: L.LightningDataModule):
        """
        Update the data module of the project.

        Args:
            datamodule (L.LightningDataModule): New data module to be used in the project.
        """
        self.datamodule = datamodule
