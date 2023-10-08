from typing import TYPE_CHECKING, Any, Dict, List

from art.utils.enums import TrainingStage

if TYPE_CHECKING:
    from art.core.base_components.base_model import ArtModule
    from art.experiment.Experiment import ArtProject
    from art.step.steps import Step


class DefaultMetric:
    pass


class DefaultModel:
    pass


class SkippedMetric:
    def __init__(
        self,
        metric,
        stages: List[str] = [TrainingStage.TRAIN.name, TrainingStage.VALIDATION.name],
    ):
        self.metric = metric.__class__.__name__
        self.stages = stages


class MetricCalculator:
    """Thanks to this preparing templates for different kinds of project will be very easy."""

    def __init__(self, experiment: "ArtProject"):
        self.metrics = []
        self.experiment = experiment

    def build_name(self: Any, model: "ArtModule", metric: Any):
        step, stage = (
            self.experiment.state.get_current_step(),
            self.experiment.state.get_current_stage(),
        )
        return f"{metric.__class__.__name__}-{model.__class__.__name__}-{stage}-{step}"

    def to(self: Any, device: str):
        self.metrics = [metric.to(device) for metric in self.metrics]

    def add_metrics(self: Any, metric: Any):
        self.metrics.extend(metric)

    def compile(self, skipped_metrics: List[SkippedMetric]):
        self.compiled_metrics = {
            TrainingStage.TRAIN.value: [],
            TrainingStage.VALIDATION.value: [],
            TrainingStage.TEST.value: [],
            TrainingStage.SANITY_CHECK.value: [],
        }

        skipped_metrics_dict = {sm.metric: sm.stages for sm in skipped_metrics}
        for metric in self.metrics:
            if hasattr(metric, "reset"):
                metric.reset()  # to make sure state is forgetten between steps
            metric_name = metric.__class__.__name__
            if metric_name not in skipped_metrics_dict.keys():
                for stage in self.compiled_metrics.keys():
                    self.compiled_metrics[stage].append(metric)
            else:
                for stage in self.compiled_metrics.keys():
                    if stage not in skipped_metrics_dict[metric_name]:
                        self.compiled_metrics[stage].append(metric)

    def __call__(self, model: "ArtModule", data_for_metrics: Dict):
        stage = self.experiment.state.get_current_stage()
        for metric in self.compiled_metrics[stage]:
            prepared_data = model.prepare_for_metric(data_for_metrics)
            metric_val = metric(*prepared_data)
            metric_name = self.build_name(model, metric)
            model.log(metric_name, metric_val, on_step=False, on_epoch=True)

            data_for_metrics[metric.__class__.__name__] = metric_val

        return data_for_metrics
