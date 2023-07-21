from enum import Enum

from art.new_structure.core.project_definitions.builders import (
    ClassificationProjectBuilder,
)


class BuildersDictionary(Enum):
    CLASSIFICATION = ClassificationProjectBuilder
    REGRESSION = None
    CLUSTERING = None
    TIME_SERIES = None
