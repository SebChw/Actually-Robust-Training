import json
from abc import ABC, abstractmethod
from pathlib import Path

BASE_PATH = Path("checkpoints")


class StepSaver(ABC):
    @abstractmethod
    def save(self, obj: any, step_id: int, step_name: str, filename: str):
        pass

    @abstractmethod
    def load(self, step_name: str, filename: str):
        pass

    def ensure_directory(self, step_id: int, step_name: str):
        return (BASE_PATH / f"{step_id}_{step_name}").mkdir(parents=True, exist_ok=True)

    def get_path(self, step_id: int, step_name: str, filename: str):
        return BASE_PATH / f"{step_id}_{step_name}" / filename


class JSONStepSaver(StepSaver):
    def save(self, obj: any, step_id: int, step_name: str, filename: str):
        self.ensure_directory(step_id, step_name)
        with open(self.get_path(step_id, step_name, filename), "w") as f:
            json.dump(obj, f)

    def load(self, step_name: str, filename: str):
        # TODO: This assumes that there will bo only one run of given check? is this ok?
        dir_path = list(BASE_PATH.glob(f"*_{step_name}"))[0]
        with open(dir_path / filename, "r") as f:
            return json.load(f)
