import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from art.core.base_components.base_model import ArtModule
from art.utils.enums import TrainingStage


@dataclass
class ResultOfCheck:
    is_positive: bool = field(default=True)
    error: str = field(default=None)


class Check(ABC):
    name: str
    description: str
    required_files: List[str]

    def __init__(
        self,
        required_key_metric,  # This requires an object which was used to calculate metric
        required_key_stage: TrainingStage,
        required_value: float,
    ):
        self.required_key_metric = required_key_metric
        self.required_key_stage = required_key_stage
        self.required_value = required_value

    @abstractmethod
    def _check_method(self, result) -> ResultOfCheck:
        pass

    def build_required_key(self, step, stage, metric):
        metric = metric.__class__.__name__
        model_name = step.get_model_name()
        step_name = step.name
        self.required_key = f"{metric}-{model_name}-{stage.name}-{step_name}"

    def check(self, step) -> ResultOfCheck:
        result = step.get_results()
        self.build_required_key(step, self.required_key_stage, self.required_key_metric)
        return self._check_method(result)


class CheckScoreExists(Check):
    def _check_method(self, result) -> ResultOfCheck:
        if self.required_key in result:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {self.required_key} is not in results.json",
            )


class CheckScoreEqualsTo(Check):
    def _check_method(self, result) -> ResultOfCheck:
        if result[self.required_key] == self.required_value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not equal to {self.required_value}",
            )


class CheckScoreCloseTo(Check):
    def __init__(
        self,
        required_key_metric,  # This requires an object which was used to calculate metric
        required_key_stage: TrainingStage,
        required_value: float,
        rel_tol: float=1e-09,
        abs_tol: float=0.0,
    ):
        super().__init__(required_key_metric, required_key_stage, required_value)
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def _check_method(self, result) -> ResultOfCheck:
        if math.isclose(
            result[self.required_key],
            self.required_value,
            rel_tol=self.rel_tol,
            abs_tol=self.abs_tol,
        ):
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not equal to {self.required_value}",
            )


class CheckScoreGreaterThan(Check):
    def _check_method(self, result) -> ResultOfCheck:
        if result[self.required_key] > self.required_value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not greater than {self.required_value}",
            )


class CheckScoreLessThan(Check):
    def _check_method(self, result) -> ResultOfCheck:
        if result[self.required_key] < self.required_value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not less than {self.required_value}",
            )
