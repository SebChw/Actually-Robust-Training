from typing import List

from art.new_structure.core.experiment.steps import Step


class Experiment:
    name: str
    steps: Step
    logger: object  # probably lightning logger
