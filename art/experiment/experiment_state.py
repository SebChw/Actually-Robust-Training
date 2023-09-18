"""Singleton that knows the current experiment state + have information about the world around"""
from typing import Union

from art.enums import TrainingStage
from art.step.step import Step


class ExperimentState:
    current_step: Union[Step, None]
    current_stage: TrainingStage = TrainingStage.TRAIN
    steps: list
    status: str
    """
    steps:{
    "model_name": [1,2,3,4,5,6],
    "model2_name: [1,2,3,4,5,6]
    
    }"""

    def __init__(self):
        self.steps = []
        self.status = "created"
        self.current_step = None
        self.current_stage = TrainingStage.TRAIN

    def get_steps(self):
        return self.steps

    def add_step(self, step):
        self.steps.append(step)

    def get_current_step(self):
        return self.current_step.name

    def get_current_stage(self):
        return self.current_stage.name
