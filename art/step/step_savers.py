import json
from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any

BASE_PATH = Path("checkpoints")


class StepSaver(ABC):
    @abstractmethod
    def save(self, obj: Any, step_id: str, step_name: str, filename: str = ""):
        pass

    @abstractmethod
    def load(self, step_id: str, step_name: str, filename: str):
        pass

    def ensure_directory(self, step_id: str, step_name: str):
        return (BASE_PATH / f"{step_id}_{step_name}").mkdir(parents=True, exist_ok=True)

    def get_path(self, step_id: str, step_name: str, filename: str):
        return BASE_PATH / f"{step_id}_{step_name}" / filename

    def exists(self, step_id: str, step_name: str, filename: str):
        return self.get_path(step_id, step_name, filename).exists()


class JSONStepSaver(StepSaver):
    RESULT_NAME = "results.json"

    def save(self, obj: Any, step_id: str, step_name: str, filename: str = RESULT_NAME):
        self.ensure_directory(step_id, step_name)
        with open(self.get_path(step_id, step_name, filename), "w") as f:
            json.dump(obj, f)

    def load(self, step_id: str, step_name: str, filename: str = RESULT_NAME):
        with open(self.get_path(step_id, step_name, filename), "r") as f:
            return json.load(f)

    def exists(self, step_id: str, step_name: str, filename: str):
        self.exists(step_id, step_name, RESULT_NAME)


class MatplotLibSaver(StepSaver):
    def save(self, obj: plt.Figure, step_id: str, step_name: str, filename: str = ""):
        self.ensure_directory(step_id, step_name)
        filepath = self.get_path(step_id, step_name, filename)
        filepath.parent.mkdir(exist_ok=True)
        obj.savefig(filepath)
        plt.close(obj)

    def load(self, step_id, step_name: str, filename: str):
        raise NotImplementedError()
