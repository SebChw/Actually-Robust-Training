from art.utils.enums import PREDICTION, TARGET, TrainingStage
import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar, Any


if TYPE_CHECKING:
    from art.experiment.Experiment import Experiment
    from art.step.steps import Step
    from art.core.base_components.base_model import ArtModule


class DefaultMetric:
    pass


class DefaultModel:
    pass


class MetricCalculator:
    """Thanks to this preparing templates for different kinds of project will be very easy."""

    prepare_registry = dict()
    exceptions = dict()
    metrics = []
    exceptions_to_be_added = []
    experiment: "Experiment"

    @classmethod
    def register_prepare(
        cls: Any,
        prepare_func: callable,
        metric_classes: List = [DefaultMetric],
        model_classes: List = [DefaultModel],
    ):
        for metric_class in metric_classes:
            first_part = metric_class.__name__
            for model_class in model_classes:
                second_part = model_class.__name__
                if first_part == second_part:
                    raise ValueError(
                        "Names of prepare classes or functions must be unique!"
                    )
                cls.prepare_registry[(first_part, second_part)] = prepare_func

    @classmethod
    def set_experiment(cls: Any, experiment: "Experiment"):
        cls.experiment = experiment

    @classmethod
    def check_if_needed(cls: Any, metric: Any):
        metric = metric.__class__.__name__
        step = cls.experiment.state.get_current_step()
        stage = cls.experiment.state.get_current_stage()
        if frozenset([metric, step, stage]) in cls.exceptions:
            return False

        return True

    @classmethod
    def register_metric(
        cls: Any,
        metric: Any,
        exception_steps: Optional[List] = None,
        exception_stages: List[str] = [TrainingStage.TRAIN.name, TrainingStage.VALIDATION.name],
    ):
        # TODO maybe we can pass list here?
        cls.metrics.append(metric)
        if exception_steps is not None:
            cls.add_exception(
                metrics=[metric], steps=exception_steps, stages=exception_stages
            )

    def register_metrics(self, metrics: List[Any]):
        for metric in metrics:
            self.register_metric(metric)

    @classmethod
    def add_exception(
        cls,
        metrics: Optional[List] = None,
        steps: Optional[List["Step"]] = None,
        stages: List[str] = [TrainingStage.TRAIN.name, TrainingStage.VALIDATION.name],
    ):
        cls.exceptions_to_be_added.append((metrics, steps, stages))

    @classmethod
    def create_exceptions(cls: Any):
        for metrics, steps, stages in cls.exceptions_to_be_added:
            if metrics is None:
                metrics = cls.metrics
            if steps is None:
                steps = cls.experiment.steps

            for metric in metrics:
                metric_name = metric.__class__.__name__
                for step in steps:
                    step_name = step.name
                    for stage in stages:
                        cls.exceptions[
                            frozenset([metric_name, step_name, stage])
                        ] = True

        cls.exceptions_to_be_added = []

    @classmethod
    def unify_type(cls: Any, x: Any):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        return x

    @classmethod
    def default_prepare(cls: Any, data: Dict):
        preds, targets = data[PREDICTION], data[TARGET]
        return cls.unify_type(preds), cls.unify_type(targets)

    @classmethod
    def get_prepare_f(cls: Any, metric: Any, model: "ArtModule"):
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
            return cls.default_prepare

    @classmethod
    def build_name(cls: Any, model: "ArtModule", metric: Any):
        step, stage = (
            cls.experiment.state.get_current_step(),
            cls.experiment.state.get_current_stage(),
        )
        return f"{metric.__class__.__name__}-{model.__class__.__name__}-{stage}-{step}"

    @classmethod
    def to(cls: Any, device: str):
        cls.metrics = [metric.to(device) for metric in cls.metrics]

    @classmethod
    def compile(cls: Any, model: "ArtModule", stage=None, step=None):
        # TODO implement this. It should be called before the stage starts
        # TODO I don't know how to pass stage here yet
        # TODO this should be set and used in __call__ instead of looking at many if statements every time.
        cls.current_metrics = []

    def __call__(self, model: "ArtModule", data_for_metrics: Dict):
        for metric in self.metrics:
            if self.check_if_needed(metric):
                # TODO Instead of this get_prepare and check if needed one should
                # TODO compile list of computable metrics before the stage runs
                prepare_f = self.get_prepare_f(metric, model)
                prepared_data = prepare_f(data_for_metrics)
                metric_val = metric(*prepared_data)
                metric_name = self.build_name(model, metric)
                model.log(metric_name, metric_val)

                data_for_metrics[metric.__class__.__name__] = metric_val

        return data_for_metrics
