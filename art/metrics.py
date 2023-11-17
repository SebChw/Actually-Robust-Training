from typing import TYPE_CHECKING, Any, Dict, List

from art.utils.enums import TrainingStage

if TYPE_CHECKING:
    from art.core import ArtModule
    from art.project import ArtProject


class DefaultMetric:
    """Placeholder for a default metric."""

    pass


class DefaultModel:
    """Placeholder for a default model."""

    pass


class SkippedMetric:
    """Represents a metric that should be skipped during certain training stages."""

    def __init__(
        self,
        metric,
        stages: List[str] = [TrainingStage.TRAIN.value, TrainingStage.VALIDATION.value],
    ):
        self.metric = metric.__class__.__name__
        self.stages = stages


class MetricCalculator:
    """
    Facilitates the management and application of metrics during different stages of training.

    This class makes preparing templates for different kinds of projects easy.
    """

    def __init__(self, experiment: "ArtProject"):
        self.metrics: List[Any] = []
        self.experiment = experiment

    def build_name(self, metric: Any) -> str:
        """
        Builds a name for the metric based on its type, current stage.

        Args:
            metric (Any): The metric being calculated.
        """
        stage = self.experiment.state.get_current_stage()

        return f"{metric.__class__.__name__}-{stage}"

    def to(self, device: str):
        """
        Move all metrics to a specified device.

        Args:
            device (str): The device to move the metrics to.
        """
        self.metrics = [metric.to(device) for metric in self.metrics]

    def add_metrics(self, metric: Any):
        """
        Add metrics to the list.

        Args:
            metric (Any): The metric to add.
        """
        self.metrics.extend(metric)

    def compile(self, skipped_metrics: List[SkippedMetric]):
        """
        Organize metrics based on stages, skipping specified ones.

        Args:
            skipped_metrics (List[SkippedMetric]): A list of SkippedMetric instances.
        """
        # Initialize a dictionary to store compiled metrics for each training stage
        self.compiled_metrics: Dict = {
            TrainingStage.TRAIN.value: [],
            TrainingStage.VALIDATION.value: [],
            TrainingStage.TEST.value: [],
            TrainingStage.SANITY_CHECK.value: [],
        }

        # Convert the list of SkippedMetric instances into a dictionary for easier access
        skipped_metrics_dict = {sm.metric: sm.stages for sm in skipped_metrics}

        # Populate compiled_metrics based on whether a metric should be included or skipped for each stage
        for metric in self.metrics:
            if hasattr(metric, "reset"):
                metric.reset()  # Ensure state is forgotten between steps

            metric_name = metric.__class__.__name__
            if metric_name not in skipped_metrics_dict.keys():
                for stage in self.compiled_metrics.keys():
                    self.compiled_metrics[stage].append(metric)
            else:
                for stage in self.compiled_metrics.keys():
                    if stage not in skipped_metrics_dict[metric_name]:
                        self.compiled_metrics[stage].append(metric)

    def __call__(self, model: "ArtModule", data_for_metrics: Dict) -> Dict:
        """
        Compute and log metrics for the current training stage.

        Args:
            model (ArtModule): The model for which the metrics are being calculated.
            data_for_metrics (Dict): The data used for calculating metrics.

        Returns:
            The data used for calculating metrics.
        """
        stage = self.experiment.state.get_current_stage()
        for metric in self.compiled_metrics[stage]:
            prepared_data = model.prepare_for_metric(data_for_metrics)
            metric_val = metric(*prepared_data)
            metric_name = self.build_name(metric)
            model.log(metric_name, metric_val, on_step=False, on_epoch=True)
            data_for_metrics[metric.__class__.__name__] = metric_val

        return data_for_metrics
