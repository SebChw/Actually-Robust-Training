from typing import List

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
        for i,(step, checks) in enumerate(zip(self.steps, self.checks)):
            if i <= self.state["last_completed_state_index"]:
                continue
            # Dependency injection so that user doesn't have to pass metric function everywhere
            print(step.name)
            step(self.state["steps"])
            for check in checks:
                result = check.check(None, step._get_saved_state())
                if not result.is_positive:
                    raise Exception(f"Check failed for step: {step.name}")
            self.state["last_completed_state_index"] = i
            self.state["steps"].append(step.get_saved_state())

        self.logger = None

