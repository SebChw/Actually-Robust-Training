from typing import List

from art.new_structure.core.experiment.steps import Step


class Experiment:
    name: str
    logger: object  # probably lightning logger
