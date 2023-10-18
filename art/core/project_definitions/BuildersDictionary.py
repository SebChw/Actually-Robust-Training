from enum import Enum

from art.core.project_definitions.builders import ClassificationProjectBuilder


class BuildersDictionary(Enum):
    """An enum (dictionary) of all available builders."""
    CLASSIFICATION = ClassificationProjectBuilder
    REGRESSION = None
    CLUSTERING = None
    TIME_SERIES = None
