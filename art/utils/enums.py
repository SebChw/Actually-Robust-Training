from enum import Enum

TARGET = "target"
PREDICTION = "prediction"
LOSS = "loss"
INPUT = "input"
BATCH = "batch"
TRAIN_LOSS = "train_loss"
VALIDATION_LOSS = "validation_loss"


class TrainingStage(Enum):
    """
    Training stage enum
    """

    TRAIN = "train"
    VALIDATION = "validate"
    TEST = "test"
    SANITY_CHECK = "sanity_check"
