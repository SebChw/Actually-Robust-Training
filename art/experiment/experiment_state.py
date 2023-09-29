"""Singleton that knows the current experiment state + have information about the world around"""
from collections import defaultdict
from typing import Dict, Union

from art.utils.enums import TrainingStage
from art.step.step import Step


class ExperimentState:
    current_step: Union[Step, None]
    current_stage: TrainingStage = TrainingStage.TRAIN
    step_states: Dict[str, Dict[str, Dict[str, str]]]
    status: str
    """
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
        self.current_stage = TrainingStage.TRAIN

    def get_steps(self):
        return self.step_states

    def add_step(self, step):
        self.step_states.append(step)

    def get_current_step(self):
        return self.current_step.name

    def get_current_stage(self):
        return self.current_step.current_stage.name
