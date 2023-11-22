import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ResultOfCheck:
    """
    Dataclass representing the result of a check operation.

    Attributes:
        is_positive (bool): Indicates if the check was successful. Defaults to True.
        error (str): Error message if the check was not successful. Defaults to an empty string.
    """

    is_positive: bool = field(default=True)
    error: str = field(default="")


class Check(ABC):
    """
    Abstract base class for defining checks.

    Attributes:
        name (str): Name of the check.
        description (str): Description of the check.
        required_files (List[str]): List of files that are required for this check.
    """

    name: str
    description: str
    required_files: List[str]

    @abstractmethod
    def check(self, step) -> ResultOfCheck:
        """
        Abstract method to execute the check on the provided step.

        Args:
            step: The step to check.

        Returns:
            ResultOfCheck: The result of the check.
        """
        pass


class CheckResult(Check):
    """
    Abstract class for checks that are based on the results of a step.
    """

    @abstractmethod
    def _check_method(self, result) -> ResultOfCheck:
        """
        Abstract method that defines the logic of the check on the result.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
        pass

    def check(self, step) -> ResultOfCheck:
        """
        Executes the check on the result of the provided step.

        Args:
            step: The step whose results are to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
        last_run = step.get_latest_run()
        result = last_run["scores"] | last_run["parameters"] | last_run
        return self._check_method(result)


class CheckResultExists(CheckResult):
    """
    Concrete check class to verify the existence of a required key in the results.

    Attributes:
        required_key (str): The key that should exist in the results.
    """

    def __init__(self, required_key):
        """
        Constructor for the CheckResultExists class.

        Args:
            required_key (str): The key that should exist in the results.
        """
        self.required_key = required_key

    def _check_method(self, result) -> ResultOfCheck:
        """
        Checks if the required_key exists in the result.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check, indicating success if the required_key exists.
        """

        if self.required_key in result:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {self.required_key} is not in results.json",
            )


class CheckScore(CheckResult):
    """
    Base class for checking scores based on a specific metric.

    Attributes:
        metric: An object used to calculate the metric.
        value (float): The expected value of the metric.
    """

    def __init__(
        self,
        metric,  # This requires an object which was used to calculate metric
        value: float,
    ):
        self.metric = metric
        self.value = value

    def build_required_key(self, step, metric):
        """
        Constructs the key for the metric based on the metric's name, the model's name,
        the current step's name, and the current check stage.

        Args:
            step: The step in which the metric was calculated.
            metric: The metric object.
        """
        metric = metric.__class__.__name__
        stage = step.get_check_stage()
        self.required_key = f"{metric}-{stage}"

    def check(self, step) -> ResultOfCheck:
        """
        Executes the check on the result of the provided step.

        Args:
            step: The step whose results are to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
        last_run = step.get_latest_run()
        result = last_run["scores"]
        self.build_required_key(step, self.metric)
        return self._check_method(result)


class CheckScoreExists(CheckScore):
    """
    Check to verify the existence of a score in the results based on a specific metric.

    Attributes:
        metric: An object used to calculate the metric.
    """

    def __init__(self, metric):
        super().__init__(metric, None)

    def _check_method(self, result) -> ResultOfCheck:
        """
        Checks if the constructed key based on the metric exists in the result.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check, indicating success if the key exists.
        """
        if self.required_key in result:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {self.required_key} is not in results.json",
            )


class CheckScoreEqualsTo(CheckScore):
    """
    Check to verify if a score in the results based on a specific metric is equal to an expected value.
    """

    def _check_method(self, result) -> ResultOfCheck:
        """
        Checks if the score associated with the constructed key is equal to the expected value.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
        if result[self.required_key] == self.value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not equal to {self.value}",
            )


class CheckScoreCloseTo(CheckScore):
    """
    Check to verify if a score in the results based on a specific metric is close to an expected value.

    Attributes:
        rel_tol (float): Relative tolerance. Defaults to 1e-09.
        abs_tol (float): Absolute tolerance. Defaults to 0.0.
    """

    def __init__(
        self,
        metric,
        value: float,
        rel_tol: float = 1e-09,
        abs_tol: float = 0.0,
    ):
        super().__init__(metric, value)
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def _check_method(self, result) -> ResultOfCheck:
        """
        Checks if the score associated with the constructed key is close to the expected value.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
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


class CheckScoreGreaterThan(CheckScore):
    """
    Check to verify if a score in the results based on a specific metric is greater than an expected value.
    """

    def _check_method(self, result) -> ResultOfCheck:
        """
        Checks if the score associated with the constructed key is greater than the expected value.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
        if result[self.required_key] > self.value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not greater than {self.value}",
            )


class CheckScoreLessThan(CheckScore):
    """
    Check to verify if a score in the results based on a specific metric is less than an expected value.
    """

    def _check_method(self, result) -> ResultOfCheck:
        """
        Checks if the score associated with the constructed key is less than the expected value.

        Args:
            result: The result to be checked.

        Returns:
            ResultOfCheck: The result of the check.
        """
        if result[self.required_key] < self.value:
            return ResultOfCheck(is_positive=True)
        else:
            return ResultOfCheck(
                is_positive=False,
                error=f"Score {result[self.required_key]} is not less than {self.value}",
            )
