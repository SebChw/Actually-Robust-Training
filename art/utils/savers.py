import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from art.utils.exceptions import SavingArtResultsException
from art.utils.paths import get_checkpoint_step_dir_path

if TYPE_CHECKING:
    from art.steps import Step


class StepSaver(ABC):
    """
    Abstract base class for saving and loading steps.
    """

    @abstractmethod
    def save(self, obj: Any, full_step_name: str, filename: str = ""):
        """
        Abstract method to save an object.

        Args:
            obj (Any): The object to save.
            full_step_name (str): The full name of the step.
            filename (str, optional): The name of the file to save to. Defaults to an empty string.
        """
        pass

    @abstractmethod
    def load(self, full_step_name: str, filename: str):
        """
        Abstract method to load an object.

        Args:
            full_step_name (str): The full name of the step.
            filename (str): The name of the file to load from.

        Returns:
            Any: Loaded object.
        """
        pass

    def ensure_directory(self, full_step_name: str):
        """
        Ensure the directory for the given step exists.

        Args:
            full_step_name (str): The full name of the step.

        Returns:
            bool: True if directory exists or is created successfully, otherwise False.
        """
        return get_checkpoint_step_dir_path(full_step_name).mkdir(
            parents=True, exist_ok=True
        )

    def get_path(self, full_step_name: str, filename: str):
        """
        Get the path for the given file of the step.

        Args:
            full_step_name (str): The full name of the step.
            filename (str): The name of the file.

        Returns:
            Path: The full path to the file.
        """
        return get_checkpoint_step_dir_path(full_step_name) / filename

    def exists(self, full_step_name: str, filename: str):
        """
        Check if a file for a given step exists.

        Args:
            full_step_name (str): The full name of the step.
            filename (str): The name of the file.

        Returns:
            bool: True if the file exists, otherwise False.
        """
        return self.get_path(full_step_name, filename).exists()


class JSONStepSaver(StepSaver):
    """
    Class to save and load steps in JSON format.
    """

    RESULT_NAME = "results.json"

    def save(self, obj: "Step", full_step_name: str, filename: str = RESULT_NAME):
        """
        Save an object as a JSON file.

        Args:
            step (Step): The step to save.
            filename (str, optional): The name of the JSON file. Defaults to "results.json".
        """

        self.ensure_directory(full_step_name)
        results_file = self.get_path(full_step_name, filename)
        if results_file.exists():
            current_results = self.load(full_step_name, filename)
        else:
            current_results = {"name": obj.name, "model": obj.model_name, "runs": []}

        current_results["runs"].insert(0, obj.results)

        for key in obj.results.keys():
            if key == "parameters":
                if "callbacks" in obj.results["parameters"].keys():
                    del obj.results["parameters"]["callbacks"]

        with open(results_file, "w") as f:
            try:
                json.dump(current_results, f)
            except TypeError as e:
                print(current_results)
                raise SavingArtResultsException(
                    f"Error while saving results for step {full_step_name}: {e}. If you have ellipsis (...) in your results, it may cause the issue."
                )

    def load(self, full_step_name, filename: str = RESULT_NAME):
        """
        Load an object from a JSON file.

        Args:
            full_step_name (str): The full name of the step.
            filename (str, optional): The name of the JSON file. Defaults to "results.json".

        Returns:
            Any: Loaded object.
        """
        with open(self.get_path(full_step_name, filename), "r") as f:
            return json.load(f)


class MatplotLibSaver(StepSaver):
    """
    Class to save figures using Matplotlib.
    """

    def save(self, obj: plt.Figure, full_step_name: str, filename: str = ""):
        """
        Save a Matplotlib figure.

        Args:
            obj (plt.Figure): The figure to save.
            full_step_name (str): The full name of the step.
            filename (str): The name of the file to save the figure to.
        """
        self.ensure_directory(full_step_name)
        filepath = self.get_path(full_step_name, filename)
        filepath.parent.mkdir(exist_ok=True)
        obj.savefig(filepath)
        plt.close(obj)

    def load(self, full_step_name: str, filename: str):
        """
        Load a Matplotlib figure. This method is not implemented.

        Args:
            full_step_name (str): The full name of the step.
            filename (str): The name of the file containing the figure.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()
