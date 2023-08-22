from abc import ABC, abstractmethod
from typing import Any, List, Dict
from art.step.step_savers import JSONStepSaver

from art.enums import TrainingStage
from art.experiment_state import ExperimentState


class Step(ABC):
    name: str
    description: str
    STEPS_REGISTRY = []#TODO I do not think this is a good idea...

    @classmethod
    def get_id(cls, instance):
        return cls.STEPS_REGISTRY.index(instance)

    def __init__(self):
        self.STEPS_REGISTRY.append(self)
        self.results = {}

    def __call__(self, previous_states: List[Dict]):
        ExperimentState.current_stage = TrainingStage.VALIDATION
        ExperimentState.current_step = self
        self.do(previous_states)
        JSONStepSaver().save(self.results, self.STEPS_REGISTRY.index(self), self.name, "results.json")

    @abstractmethod
    def do(self, previous_states: List[Dict]):
        pass

    def _get_saved_state(self) -> Dict[str, str]:
        return_dict = {"results": "results.json"}
        return_dict.update(self.get_saved_state())
        return return_dict

    def get_saved_state(self) -> Dict[str, str]:
        return {}
