from typing import TYPE_CHECKING, List

from art.experiment.experiment_state import ExperimentState
from art.metric_calculator import MetricCalculator
from art.step.checks import Check

if TYPE_CHECKING:
    from art.step.step import Step


class Experiment:
    name: str
    steps: List["Step"]
    logger: object  # probably lightning logger
    state: ExperimentState

    def __init__(self, name, **kwargs):
        # Potentially we can save file versions to show which model etc was used.
        self.name = name
        self.steps = []
        self.checks = []
        self.state = ExperimentState()
        # self.update_dashboard(self.steps) # now from each step we take internal information it has remembered and save them to show on a dashboard

    def add_step(self, step: "Step", checks: List[Check]):
        self.steps.append(step)
        step.set_step_id(len(self.steps))
        self.checks.append(checks)

    def fill_step_states(self, step):
        self.state.step_states[step.get_model_name()][
            step.get_name_with_id()
        ] = step.get_results()

    def run_all(self):
        MetricCalculator.set_experiment(self)
        MetricCalculator.create_exceptions()

        for step, checks in zip(self.steps, self.checks):
            self.state.current_step = step

            step_passed = True
            if not step.was_run():  # fmt: skip
                step_passed = False
            else:
                step.load_results()
                for check in checks:
                    check_result = check.check(step)
                    if not check_result.is_positive:
                        step_passed = False
                        break

            if step_passed:
                step_current_hash = step.get_hash()
                step_saved_hash = step.get_results()["hash"]
                model_changed = False
                if step_current_hash != step_saved_hash:
                    model_changed = True
                if not model_changed:
                    print(
                        f"Step {step.name}_{step.get_step_id()} was already completed."
                    )
                    self.fill_step_states(step)
                    continue
                else:
                    print(
                        f"We've noticed you changed some code in your model in step {step.name}_{step.get_step_id()}. We will rerun it."
                    )

            step.add_result("hash", step.get_hash())

            step(self.state.step_states)
            for check in checks:
                result = check.check(step)
                if not result.is_positive:
                    raise Exception(
                        f"Check failed for step: {step.name}. Reason: {result.error}"
                    )

            self.fill_step_states(step)

        self.logger = None

    def get_steps(self):
        return self.steps

    def exchange_step(self, step: "Step", step_id=-1):
        self.steps[step_id] = step
        if step_id == -1:
            step_id = len(self.steps)
        step.set_step_id(step_id)
