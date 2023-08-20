import json
from abc import ABC
from pathlib import Path

BASE_PATH = Path("checkpoints")


class StepSaver(ABC):
    def save(self, obj:any, step_name: str, filename: str):
        pass

    def load(self, step_name: str, filename: str):
        pass


class JSONStepSaver(StepSaver):
    def save(self, obj:any, step_name: str, filename: str):
        (BASE_PATH/step_name).mkdir(parents=True, exist_ok=True)
        with open(BASE_PATH/step_name/filename, "w") as f:
            json.dump(obj, f)

    def load(self, step_name: str, filename: str):
        with open(BASE_PATH/step_name/filename, "r") as f:
            return json.load(f)
