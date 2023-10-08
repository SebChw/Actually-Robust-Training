from typing import TYPE_CHECKING, Any, Dict, List, Optional

import lightning as L

from art.core.MetricCalculator import MetricCalculator, SkippedMetric
from art.experiment.experiment_state import ArtProjectState
from art.step.checks import Check

if TYPE_CHECKING:
    from art.step.step import Step


class ArtProject:
    name: str
    steps: List
    logger: object  # probably lightning logger
    state: ArtProjectState

    def __init__(self, name, datamodule: L.LightningDataModule, **kwargs):
        # Potentially we can save file versions to show which model etc was used.
        self.name = name
        self.steps = []
        self.datamodule = datamodule
        self.state = ArtProjectState()
        self.metric_calculator = MetricCalculator(self)
        # self.update_dashboard(self.steps) # now from each step we take internal information it has remembered and save them to show on a dashboard

    def add_step(
        self,
        step: "Step",
        checks: List[Check],
        skipped_metrics: List[SkippedMetric] = [],
    ):
        self.steps.append(
            {
                "step": step,
                "checks": checks,
                "skipped_metrics": skipped_metrics,
            }
        )
        step.set_step_id(len(self.steps))

    def fill_step_states(self, step: "Step"):
        self.state.step_states[step.get_model_name()][
            step.get_name_with_id()
        ] = step.get_results()

    def check_checks(self, step: "Step", checks: List[Check]):
        for check in checks:
            result = check.check(step)
            if not result.is_positive:
                raise Exception(
                    f"Check failed for step: {step.name}. Reason: {result.error}"
                )

    def check_if_must_be_run(self, step: "Step", checks: List[Check]):
        if not step.was_run():
            return True
        else:
            step.load_results()
            try:
                self.check_checks(step, checks)
            except Exception as e:
                return True

        step_current_hash = step.get_hash()
        step_saved_hash = step.get_results()["hash"]
        model_changed = True if step_current_hash != step_saved_hash else False

        if model_changed:
            print(
                f"Code of the model in {step.get_full_step_name()} was changed. Rerun needed."
            )
            return True

        print(f"Step {step.name}_{step.get_step_id()} was already completed.")
        return False

    def run_all(self):
        for step in self.steps:
            self.metric_calculator.compile(step["skipped_metrics"])
            step, checks = step["step"], step["checks"]

            self.state.current_step = step

            if not self.check_if_must_be_run(step, checks):
                self.fill_step_states(step)
                continue

            step.add_result("hash", step.get_hash())

            step(self.state.step_states, self.datamodule, self.metric_calculator)

            self.check_checks(step, checks)
            self.fill_step_states(step)

    def get_steps(self):
        return self.steps

    def get_step(self, step_id):
        return self.steps[step_id]["step"]


    def replace_step(self, step: "Step", step_id=-1):
        self.steps[step_id]["step"] = step
        if step_id == -1:
            step_id = len(self.steps)
        step.set_step_id(step_id)

    def register_metrics(
        self: Any,
        metrics: List[Any],
    ):
        self.metric_calculator.add_metrics(metrics)

    def to(self: Any, device: str):
        self.metric_calculator.to(device)

    def update_datamodule(self, datamodule: L.LightningDataModule):
        self.datamodule = datamodule
