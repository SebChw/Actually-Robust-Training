from enum import Enum

TARGET = "target"
PREDICTION = "prediction"
LOSS = "loss"
INPUT = "input"
BATCH = "batch"
TRAIN_LOSS = "train_loss"
VALIDATION_LOSS = "validation_loss"


class TrainingStage(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
