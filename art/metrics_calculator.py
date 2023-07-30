from typing import Dict


class MetricCalculator:
    """Thanks to this preparing templates for different kinds of project will be very easy."""

    prepare_registry = {}

    @classmethod
    def register_prepare(cls, metric_class=None, model_class=None):
        def decorator(prepare_func):
            if metric_class is None and model_class is None:
                cls.prepare_registry["default"] = prepare_func

            elif metric_class is not None and model_class is not None:
                cls.prepare_registry[
                    (metric_class.__name__, model_class.__name__)
                ] = prepare_func

            elif metric_class is not None:
                cls.prepare_registry[metric_class.__name__] = prepare_func

            elif model_class is not None:
                cls.prepare_registry[model_class.__name__] = prepare_func

        return decorator

    def check_if_needed(self, metric, lightning_module, stage: str):
        metric_name = metric.__class__.__name__
        l_module_name = lightning_module.__class__.__name__
        if metric_name in self.exceptions:
            return True

        if (metric_name, l_module_name) in self.exceptions[metric_name]:
            return True

        if (metric_name, l_module_name, stage) in self.exceptions[metric_name]:
            return True

        return False

    def get_prepare_f(self, metric, lightning_module):
        metric_class = metric.__class__.__name__
        model_class = lightning_module.__class__.__name__

        if (metric_class, model_class) in self.prepare_registry:
            return self.prepare_registry[(metric_class, model_class)]

        elif metric_class in self.prepare_registry:
            return self.prepare_registry[metric_class]

        elif model_class in self.prepare_registry:
            return self.prepare_registry[model_class]

        elif "default" in self.prepare_registry:
            return self.prepare_registry["default"]

        else:
            lambda y: y

    def __init__(self):
        self.metrics = []
        # In this dictionary we define when metric shouldn't be calculated
        # Exceptions are defined by user before everything starts.
        self.exceptions = {}

    def calculate_metrics(self, lightning_module, data_for_metrics: Dict, stage: str):
        for metric in self.metrics:
            if self.check_if_needed(metric, lightning_module, stage):
                prepare_f = self.get_prepare_f(metric, lightning_module)
                # The last question would be how to know in which stage we are to log separately for Overfit and OverfitOneBatch etc.
                # + should we somehow store all previously calculated data.
                lightning_module.log(metric(prepare_f(data_for_metrics)))
