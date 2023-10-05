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
        metric,  # This requires an object which was used to calculate metric
        stage: TrainingStage,
        value: float,
    ):
        self.metric = metric
        self.stage = stage
        self.value = value

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
        self.build_required_key(step, self.stage, self.metric)
        return self._check_method(result)


class CheckScoreExists(Check):
    def __init__(self, metric, stage: TrainingStage):
        super().__init__(metric, stage, None)

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
        if result[self.required_key] == self.value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not equal to {self.value}",
            )


class CheckScoreCloseTo(Check):
    def __init__(
        self,
        metric,  # This requires an object which was used to calculate metric
        stage: TrainingStage,
        value: float,
        rel_tol: float = 1e-09,
        abs_tol: float = 0.0,
    ):
        super().__init__(metric, stage, value)
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def _check_method(self, result) -> ResultOfCheck:
        if math.isclose(
            result[self.required_key],
            self.value,
            rel_tol=self.rel_tol,
            abs_tol=self.abs_tol,
        ):
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not equal to {self.value}",
            )


class CheckScoreGreaterThan(Check):
    def _check_method(self, result) -> ResultOfCheck:
        if result[self.required_key] > self.value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not greater than {self.value}",
            )


class CheckScoreLessThan(Check):
    def _check_method(self, result) -> ResultOfCheck:
        if result[self.required_key] < self.value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not less than {self.value}",
            )
