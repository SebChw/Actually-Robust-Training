from typing import List, TYPE_CHECKING

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
        self.state.step_states[step.get_model_name()][step.get_name_with_id()] = step._get_saved_state()
        step.set_step_id(len(self.steps))
        step.set_experiment(self)
        self.checks.append(checks)

    def run_all(self):
        MetricCalculator.set_experiment(self)
        MetricCalculator.create_exceptions()
        for step, checks in zip(self.steps, self.checks):
            self.state.current_step = step
            step_passed = True

            for check in checks:
                result = check.check(step)
                if not result.is_positive:
                    step_passed = False
                    break

            # TODO implement step changed utility
            # step_changed = False
            # if step_changed_utility(step):
            #   step_changed = True

            if step_passed:
                print(f"Step {step.name}_{step.get_step_id()} was already completed.")
                continue

            step(self.state.step_states)
            for check in checks:
                result = check.check(step)
                if not result.is_positive:
                    raise Exception(f"Check failed for step: {step.name}. Reason: {result.error}")
        self.logger = None
