from typing import Dict, List, Optional

from art.core.experiment.step.steps import Step


class DefaultMetric:
    pass


class DefaultModel:
    pass


class MetricCalculator:
    """Thanks to this preparing templates for different kinds of project will be very easy."""

    prepare_registry = dict()
    exceptions = dict()
    metrics = []

    @classmethod
    def register_prepare(
        cls,
        prepare_func,
        metric_classes: List = [DefaultMetric],
        model_classes: List = [DefaultModel],
    ):
        for metric_class in metric_classes:
            first_part = metric_class.__class__.__name__
            for model_class in model_classes:
                second_part = model_class.__class__.__name__
                if first_part == second_part:
                    raise ValueError(
                        "Names of prepare classes or functions must be unique!"
                    )
                cls.prepare_registry[(first_part, second_part)] = prepare_func

    @classmethod
    def check_if_needed(cls, metric, step: str, stage):
        metric = metric.__class__.__name__
        if frozenset([metric, step, stage]) in cls.exceptions:
            return True

        return False

    @classmethod
    def add_exception(
        cls,
        metrics: Optional[List] = None,
        steps: Optional[List[Step]] = None,
        stages: List[str] = ["train", "val"],
    ):
        if metrics is None:
            metrics = cls.metrics
        if steps is None:
            steps = Step.STEPS_REGISTRY

        for metric in metrics:
            metric_name = metric.__class__.__name__
            for step in steps:
                step_name = step.name
                for stage in stages:
                    cls.exceptions[frozenset([metric_name, step_name, stage])] = True

    @classmethod
    def get_prepare_f(cls, metric, model):
        metric_name = metric.__class__.__name__
        model_name = model.__class__.__name__

        if (metric_name, model_name) in cls.prepare_registry:
            return cls.prepare_registry[(metric_name, model_name)]
        elif (metric_name, DefaultModel.__name__) in cls.prepare_registry:
            return cls.prepare_registry[(metric_name, DefaultModel.__name__)]
        elif (DefaultMetric.__name__, model_name) in cls.prepare_registry:
            return cls.prepare_registry[(DefaultMetric.__name__, model_name)]
        elif (DefaultMetric.__name__, DefaultModel.__name__) in cls.prepare_registry:
            return cls.prepare_registry[(DefaultMetric.__name__, DefaultModel.__name__)]
        else:
            return lambda y: y

    @classmethod
    def calculate_metrics(cls, model, data_for_metrics: Dict, stage: str, step):
        for metric in cls.metrics:
            if cls.check_if_needed(metric, step, stage):
                prepare_f = cls.get_prepare_f(metric, model)
                model.log(metric(prepare_f(data_for_metrics)))
