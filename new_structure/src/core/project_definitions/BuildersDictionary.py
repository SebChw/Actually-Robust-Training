from enum import Enum
from src.core.project_definitions.builders import ClassificationProjectBuilder


class BuildersDictionary(Enum):
    CLASSIFICATION = ClassificationProjectBuilder
    REGRESSION = None
    CLUSTERING = None
    TIME_SERIES = None
