from typing import List

from new_structure.src.core.experiment.steps import Step


class Experiment:
    name: str
    steps: Step
    logger: object  # probably lightning logger
