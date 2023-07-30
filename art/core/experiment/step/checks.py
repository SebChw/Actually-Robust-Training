from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from core.experiment.step.step_savers import JSONStepSaver


@dataclass
class ResultOfCheck:
    is_positive: bool = field(default=True)
    error: str = field(default=None)


class Check(ABC):
    name: str
    description: str
    required_files: List[str]

    def __init__(self, name: str, description: str, required_files: List[str]):
        self.name = name
        self.description = description
        self.required_files = required_files

    @abstractmethod
    def check(self, dataset, step_state_dict: Dict[str, str]) -> ResultOfCheck:
        assert all([file in step_state_dict for file in self.required_files])



class CheckScoreExists(Check):
    def __init__(self, name: str, description: str, score_filed: str):
        super().__init__(name, description, ["results"])
        self.score_filed = score_filed


    def check(self, dataset, step_state_dict: Dict[str, str]) -> ResultOfCheck:
        super().check(dataset, step_state_dict)
        result = JSONStepSaver().load(self.name, step_state_dict["results"])
        if self.score_filed in result:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(is_positive=False, error=f"Score {self.score_filed} is not in results.json")

class CheckScoreEqualsTo(Check):
    def __init__(self, name: str, description: str, score_filed: str, score: float):
        super().__init__(name, description, ["results"])
        self.score_filed = score_filed
        self.score = score


    def check(self, dataset, step_state_dict: Dict[str, str]) -> ResultOfCheck:
        super().check(dataset, step_state_dict)
        result = JSONStepSaver().load(self.name, step_state_dict["results"])
        if result[self.score_filed] == self.score:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(is_positive=False, error=f"Score {result[self.score_filed]} is not equal to {self.score}")


class CheckScoreGreaterThan(Check):
    def __init__(self, name: str, description: str, score_filed: str, score: float):
        super().__init__(name, description, ["results"])
        self.score_filed = score_filed
        self.score = score


    def check(self, dataset, step_state_dict: Dict[str, str]) -> ResultOfCheck:
        super().check(dataset, step_state_dict)
        result = JSONStepSaver().load(self.name, step_state_dict["results"])
        if result[self.score_filed] > self.score:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(is_positive=False, error=f"Score {result[self.score_filed]} is not greater than {self.score}")



class CheckScoreLessThan(Check):
    def __init__(self, name: str, description: str, score_filed: str, score: float):
        super().__init__(name, description, ["results"])
        self.score_filed = score_filed
        self.score = score


    def check(self, dataset, step_state_dict: Dict[str, str]) -> ResultOfCheck:
        super().check(dataset, step_state_dict)
        result = JSONStepSaver().load(self.name, step_state_dict["results"])
        if result[self.score_filed] < self.score:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(is_positive=False, error=f"Score {result[self.score_filed]} is not less than {self.score}")
