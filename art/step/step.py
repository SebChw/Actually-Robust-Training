from abc import ABC, abstractmethod
from typing import Any, Dict, List

from art.enums import TrainingStage
from art.experiment_state import ExperimentState
from art.step.step_savers import JSONStepSaver


class Step(ABC):
    name: str
    description: str

    def __init__(self):
        self.results = {}
        self.model = None
        self.datamodule = None

    def __call__(self, previous_states: List[Dict]):
        ExperimentState.current_stage = TrainingStage.TRAIN
        ExperimentState.current_step = self
        self.do(previous_states)

        JSONStepSaver().save(
            self.results, self.get_step_id(), self.name, "results.json"
        )

    @abstractmethod
    def do(self, previous_states: List[Dict]):
        pass

    def _get_saved_state(self) -> Dict[str, str]:
        return_dict = {"results": "results.json"}
        return_dict.update(self.get_saved_state())
        return return_dict

    def get_saved_state(self) -> Dict[str, str]:
        return {}

    def get_step_id(self) -> str:
        return f"{self.model.__class__.__name__}_{self.datamodule.__class__.__name__}"
