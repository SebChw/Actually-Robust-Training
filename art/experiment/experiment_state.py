from collections import defaultdict
from typing import Dict, Union

from art.step.step import Step
from art.utils.enums import TrainingStage


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
