"""Singleton that knows the current experiment state + have information about the world around"""
from art.enums import TrainingStage


class ExperimentState:
    current_step = None
    current_stage: TrainingStage = TrainingStage.TRAIN

    @classmethod
    def get_step(cls):
        return cls.current_step.name

    @classmethod
    def get_stage(cls):
        return cls.current_stage.name
