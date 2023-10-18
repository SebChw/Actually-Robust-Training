from typing import TYPE_CHECKING, Any, List, Dict

from collections import defaultdict
from dataclasses import dataclass

import lightning as L

from art.core.MetricCalculator import MetricCalculator, SkippedMetric
from art.experiment.experiment_state import ArtProjectState
from art.step.checks import Check

import json

if TYPE_CHECKING:
    from art.step.step import Step


@dataclass
class StepStatus:
    status: str
    results: Dict[str, Any]

    def __repr__(self):
        result_repr = "\n".join(f"\t{k}: {v}" for k, v in self.results.items() if k != 'hash')
        return f"{self.status}. Results:\n{result_repr}"

    def to_dict(self):
        return {"status": self.status, "results": self.results}


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
        ] = step.get_results()

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
                raise Exception(
                    f"Check failed for step: {step.name}. Reason: {result.error}"
                )

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
            step.load_results()
            try:
                self.check_checks(step, checks)
            except Exception as e:
                print(e)
                return True

        step_current_hash = step.get_hash()
        step_saved_hash = step.get_results()["hash"]
        model_changed = True if step_current_hash != step_saved_hash else False

        if model_changed:
            print(
                f"Code of the model in {step.get_full_step_name()} was changed. Rerun needed."
            )
            return True

        return False

    def save_state(self, status: dict, path: str = 'checkpoints/state.json'):
        with open(path, 'w') as f:
            serializable_steps_status = {k: v.to_dict() for k, v in status.items()}
            json.dump(serializable_steps_status, f)

    def run_all(self, force_rerun=False):
        """
            Execute all steps in the project.

            Args:
                force_rerun (bool): Whether to force rerun all steps.
        """
        steps_status = defaultdict(lambda: StepStatus("Not run", None))

        for step in self.steps:
            self.metric_calculator.compile(step["skipped_metrics"])
            step, checks = step["step"], step["checks"]

            self.state.current_step = step

            if not self.check_if_must_be_run(step, checks) and not force_rerun:
                steps_status[step.get_full_step_name()] = StepStatus("Skipped", step.get_results())
                self.fill_step_states(step)
                self.save_state(steps_status)
                continue

            step.add_result("hash", step.get_hash())

            try:
                step(self.state.step_states, self.datamodule, self.metric_calculator)
                self.check_checks(step, checks)
                steps_status[step.get_full_step_name()] = StepStatus("Completed", step.get_results())
            except Exception as e:
                steps_status[step.get_full_step_name()] = StepStatus("Failed", step.get_results())
                self.fill_step_states(step)
                self.save_state(steps_status)
                exception_msg = f"Step {step.get_full_step_name()} failed: {e}"
                exception_msg += "\n\nSteps status:"
                for step_name, step_status in steps_status.items():
                    exception_msg += f"\n{step_name}: {step_status}"
                raise Exception(exception_msg)

            self.fill_step_states(step)

        print("Steps status:")
        for step_name, step_status in steps_status.items():
            print(f"{step_name}: {step_status}")
        self.save_state(steps_status)

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
