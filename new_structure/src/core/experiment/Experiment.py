from typing import List

from src.core.experiment.Step import Step


class Experiment:
    name: str
    steps: Step
    logger: object#probably lightning logger


