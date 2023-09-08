import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from art.enums import TrainingStage
from art.step.step_savers import JSONStepSaver


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
        name: str,
        description: str,
        required_files: List[str],
        required_key_metric,  # This requires an object which was used to calculate metric
        required_key_stage: TrainingStage,
        required_value: float,
    ):
        self.check_name = name
        self.description = description
        self.required_files = required_files
        self.required_key_metric = required_key_metric
        self.required_key_stage = required_key_stage
        self.required_value = required_value

    @abstractmethod
    def _check_method(self, result) -> ResultOfCheck:
        pass

    def build_required_key(self, step, stage, metric):
        metric = metric.__class__.__name__
        model = step.model.__class__.__name__ if hasattr(step, "model") else ""
        step_name = step.name
        self.required_key = f"{metric}-{model}-{stage}-{step_name}"

    def check(self, dataset, step) -> ResultOfCheck:
        # TODO why we pass dataset here
        step_state_dict = step._get_saved_state()
        self.build_required_key(step, self.required_key_stage, self.required_key_metric)
        assert all([file in step_state_dict for file in self.required_files])
        result = JSONStepSaver().load(step.id, step.name, step_state_dict["results"])
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
        name: str,
        description: str,
        required_key: str,
        required_value: float,
        rel_tol=1e-09,
        abs_tol=0.0,
    ):
        super().__init__(name, description, ["results"], required_key, required_value)
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
