import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import torch

from art.core.base_components.base_model import ArtModule

BASE_PATH = Path("checkpoints")


class StepSaver(ABC):
    """
    Abstract base class for saving and loading steps.
    """

    @abstractmethod
    def save(self, obj: Any, step_id: str, step_name: str, filename: str = ""):
        """
        Abstract method to save an object.

        Args:
            obj (Any): The object to save.
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str, optional): The name of the file to save to. Defaults to an empty string.
        """
        pass

    @abstractmethod
    def load(self, step_id: str, step_name: str, filename: str):
        """
        Abstract method to load an object.

        Args:
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str): The name of the file to load from.

        Returns:
            Any: Loaded object.
        """
        pass

    def ensure_directory(self, step_id: str, step_name: str):
        """
        Ensure the directory for the given step exists.

        Args:
            step_id (str): The ID of the step.
            step_name (str): The name of the step.

        Returns:
            bool: True if directory exists or is created successfully, otherwise False.
        """
        return (BASE_PATH / f"{step_id}_{step_name}").mkdir(parents=True, exist_ok=True)

    def get_path(self, step_id: str, step_name: str, filename: str):
        """
        Get the path for the given file of the step.

        Args:
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str): The name of the file.

        Returns:
            Path: The full path to the file.
        """
        return BASE_PATH / f"{step_id}_{step_name}" / filename

    def exists(self, step_id: str, step_name: str, filename: str):
        """
        Check if a file for a given step exists.

        Args:
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str): The name of the file.

        Returns:
            bool: True if the file exists, otherwise False.
        """
        return self.get_path(step_id, step_name, filename).exists()


class JSONStepSaver(StepSaver):
    """
    Class to save and load steps in JSON format.
    """

    RESULT_NAME = "results.json"

    def save(self, step: "Step", filename: str = RESULT_NAME):
        """
        Save an object as a JSON file.

        Args:
            step (Step): The step to save.
            filename (str, optional): The name of the JSON file. Defaults to "results.json".
        """
        step_id = step.get_step_id()
        step_name = step.name

        self.ensure_directory(step_id, step_name)
        results_file = self.get_path(step_id, step_name, filename)
        if results_file.exists():
            current_results = self.load(step_id, step_name, filename)
        else:
            current_results = {"name": step_name, "model": step.model_name, "runs": []}

        current_results["runs"].insert(0, step.results)

        with open(results_file, "w") as f:
            json.dump(current_results, f)

    def load(self, step_id: str, step_name: str, filename: str = RESULT_NAME):
        """
        Load an object from a JSON file.

        Args:
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str, optional): The name of the JSON file. Defaults to "results.json".

        Returns:
            Any: Loaded object.
        """
        with open(self.get_path(step_id, step_name, filename), "r") as f:
            return json.load(f)


class MatplotLibSaver(StepSaver):
    """
    Class to save figures using Matplotlib.
    """

    def save(self, obj: plt.Figure, step_id: str, step_name: str, filename: str = ""):
        """
        Save a Matplotlib figure.

        Args:
            obj (plt.Figure): The figure to save.
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str): The name of the file to save the figure to.
        """
        self.ensure_directory(step_id, step_name)
        filepath = self.get_path(step_id, step_name, filename)
        filepath.parent.mkdir(exist_ok=True)
        obj.savefig(filepath)
        plt.close(obj)

    def load(self, step_id, step_name: str, filename: str):
        """
        Load a Matplotlib figure. This method is not implemented.

        Args:
            step_id (str): The ID of the step.
            step_name (str): The name of the step.
            filename (str): The name of the file containing the figure.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()
