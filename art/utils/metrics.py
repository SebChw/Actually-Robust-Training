import torch.nn as nn


def get_metric_array(metric_class, **metric_class_kwargs):
    N_TRAIN_STAGES = 3
    return nn.ModuleList(
        [metric_class(**metric_class_kwargs) for _ in range(N_TRAIN_STAGES)]
    )
