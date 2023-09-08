from typing import List

from art.metric_calculator import MetricCalculator
from art.step.checks import Check
from art.step.step import Step


class Experiment:
    name: str
    steps: List[Step]
    logger: object  # probably lightning logger
    state: dict

    def __init__(self, name, **kwargs):
        # Potentially we can save file versions to show which model etc was used.
        # TODO, do we want to use list or something different like self.add_step(). Consider builder pattern.
        self.name = name
        self.steps = []
        self.checks = []
        #TODO think about merging it with ExperimentState class
        self.state ={
            "status": "created",
            "last_completed_state_index": -1,
            "steps": []
        }
        # self.update_dashboard(self.steps) # now from each step we take internal information it has remembered and save them to show on a dashboard

    def add_step(self, step: Step, checks: List[Check]):
        self.steps.append(step)
        self.checks.append(checks)

    def run_all(self):
        MetricCalculator.create_exceptions(self.steps)
                continue
            print(step.name)
            step(self.state["steps"])
            for check in checks:
                check.name = step.name # TODO this is solution just for now
                result = check.check(None, step._get_saved_state())
                if not result.is_positive:
                    raise Exception(f"Check failed for step: {step.name}")
            self.state["last_completed_state_index"] = i
            self.state["steps"].append(step.get_saved_state())

        self.logger = None

