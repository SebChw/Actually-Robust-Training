from abc import ABC, abstractmethod
from typing import Any, Dict

import lightning as L

from art.core.base_components.base_model import ArtModule
from art.core.MetricCalculator import MetricCalculator
from art.step.step_savers import JSONStepSaver
from art.utils.enums import TrainingStage


class Step(ABC):
    name: str
    description: str
    idx: int = None

    def __init__(self, model: ArtModule, trainer: L.Trainer):
        self.results = {}
        self.model = model
        self.trainer = trainer
        self.results = {}

    def __call__(
        self,
        previous_states: Dict,
        datamodule: L.LightningDataModule,
        metric_calculator: MetricCalculator,
    ):
        self.datamodule = datamodule
        self.model.set_metric_calculator(metric_calculator)
        self.do(previous_states)
        JSONStepSaver().save(
            self.results, self.get_step_id(), self.name, "results.json"
        )

    @abstractmethod
    def do(self, previous_states: Dict):
        pass

    def train(self, trainer_kwargs: Dict):
        self.trainer.fit(model=self.model, **trainer_kwargs)
        logged_metrics = {k: v.item() for k, v in self.trainer.logged_metrics.items()}
        self.results.update(logged_metrics)

    def validate(self, trainer_kwargs: Dict):
        result = self.trainer.validate(model=self.model, **trainer_kwargs)
        self.results.update(result[0])

    def test(self, trainer_kwargs: Dict):
        result = self.trainer.test(model=self.model, **trainer_kwargs)
        self.results.update(result[0])

    def set_step_id(self, idx: int):
        self.idx = idx

    def get_model_name(self) -> str:
        return self.model.__class__.__name__

    def get_step_id(self) -> str:
        return (
            f"{self.get_model_name()}_{self.idx}"
            if self.get_model_name() != ""
            else f"{self.idx}"
        )

    def get_name_with_id(self) -> str:
        return f"{self.idx}_{self.name}"

    def get_full_step_name(self) -> str:
        return f"{self.get_step_id()}_{self.name}"

    def get_hash(self) -> str:
        return self.model.get_hash()

    def add_result(self, name: str, value: Any):
        self.results[name] = value

    def get_results(self) -> Dict:
        return self.results

    def load_results(self):
        self.results = JSONStepSaver().load(self.get_step_id(), self.name)

    def get_current_stage(self) -> str:
        return self.trainer.state.stage.value

    def was_run(self):
        path = JSONStepSaver().get_path(
            self.get_step_id(), self.name, JSONStepSaver.RESULT_NAME
        )
        return path.exists()

    def get_check_stage(self) -> str:
        return TrainingStage.VALIDATION.value
