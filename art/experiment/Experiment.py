from typing import TYPE_CHECKING, Any, List

import lightning as L

from art.core.exceptions import CheckFailedException
from art.core.MetricCalculator import MetricCalculator, SkippedMetric
from art.experiment.experiment_state import ArtProjectState
from art.step.checks import Check

if TYPE_CHECKING:
    from art.step.step import Step


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
        self.steps = []
        self.datamodule = datamodule
        self.state = ArtProjectState()
        self.metric_calculator = MetricCalculator(self)

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
        step.set_step_id(len(self.steps))

    def fill_step_states(self, step: "Step"):
        """
        Update step states with the results from the given step.

        Args:
            step (Step): The step whose results need to be recorded.
        """
        self.state.step_states[step.get_model_name()][
            step.get_name_with_id()
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
                raise CheckFailedException(
                    f"Check failed for step: {step.name}. Reason: {result.error}"
                )
        step.set_succesfull()

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
            try:
                self.check_checks(step, checks)
            except Exception as e:
                print(e)
                return True

        step_current_hash = step.get_hash()
        step_saved_hash = step.get_latest_run()["hash"]
        model_changed = True if step_current_hash != step_saved_hash else False

        if model_changed:
            print(
                f"Code of the model in {step.get_full_step_name()} was changed. Rerun needed."
            )
            return True

        return False

    def run_all(self, force_rerun=False):
        """
        Execute all steps in the project.

        Args:
            force_rerun (bool): Whether to force rerun all steps.
        """
        for step in self.steps:
            self.metric_calculator.compile(step["skipped_metrics"])
            step, checks = step["step"], step["checks"]
            self.state.current_step = step

            if not self.check_if_must_be_run(step, checks) and not force_rerun:
                self.fill_step_states(step)
                continue
            try:
                step(self.state.step_states, self.datamodule, self.metric_calculator)
                self.check_checks(step, checks)
            except CheckFailedException as e:
                print(f"\n\n{e}\n\n")
                step.save_to_disk()
                break

            self.fill_step_states(step)
            step.save_to_disk()

        self.print_summary()

    def print_summary(self):
        """
        Prints a summary of the project.
        """
        print("Summary: ")
        for step in self.steps:
            print(step["step"])
            if not step["step"].is_succesfull():
                break

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
        if step_id == -1:
            step_id = len(self.steps)
        step.set_step_id(step_id)

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
