from abc import ABC, abstractmethod
from typing import Any, Dict, List

from art.enums import TrainingStage
from art.step.step_savers import JSONStepSaver


class Step(ABC):
    name: str
    description: str
    experiment = None
    idx: int = None

    def __init__(self):
        self.results = {}
        self.model = None
        self.datamodule = None
        self.experiment = None

    def set_experiment(self, experiment):
        self.experiment = experiment

    def __call__(self, previous_states: List[Dict]):
        self.experiment.current_stage = TrainingStage.TRAIN
        self.experiment.current_step = self
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

    def set_step_id(self, idx: int):
        self.idx = idx

    def get_model_name(self) -> str:
        return self.model.__class__.__name__

    def get_step_id(self) -> str:
        return f"{self.get_model_name()}_{self.idx}" if self.get_model_name() != "" else f"{self.idx}"

    def get_name_with_id(self) -> str:
        return f"{self.idx}_{self.name}"

    def get_full_step_name(self) -> str:
        return f"{self.get_step_id()}_{self.name}"
