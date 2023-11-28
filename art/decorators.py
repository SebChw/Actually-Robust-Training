from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from torchvision.utils import save_image

from art.loggers import art_logger, supress_stdout
from art.utils.enums import INPUT, PREDICTION, TARGET

"""
Idea is as follows:
- Instead of thid visualize_img_on_input , we can give class with a state there.
- We don't need to restrict ourselves to just visualizations, we can store information about prediction dynamics, log mean values of tensors etc.
- State of this object is updated either by user on every call and/or by lightning module (set which epoch we at etc.)
"""


def art_decorate_single_func(
    visualizing_function_in=None, visualizing_function_out=None
):
    """
    Decorates input and output of a function.

    Args:
        function_in (function, optional): Function applied on the input. Defaults to None.
        function_out (function, optional): Function applied on the output. Defaults to None.

    Returns:
        function: Decorated function.
    """

    def decorator(func):
        """
        Decorator

        Args:
            func (function): Function to decorate.
        """

        def wrapper(*args, **kwargs):
            """
            Wrapper

            Returns:
                function: Decorated function.
            """
            if visualizing_function_in is not None:
                visualizing_function_in(*args, **kwargs)
            output = func(*args, **kwargs)
            if visualizing_function_out is not None:
                visualizing_function_out(output)
            return output

        return wrapper

    return decorator


@dataclass
class ModelDecorator:
    funcion_name: str
    input_decorator: Optional[Callable] = None
    output_decorator: Optional[Callable] = None


def art_decorate(
    functions: List[Tuple[object, str]],
    input_decorator: Optional[Callable] = None,
    output_decorator: Optional[Callable] = None,
):
    """
    Decorates list of objects functions. It doesn't modify output of a function
    put can be used for logging additional information during training.

    Args:
        functions (List[Tuple[object, str]]): List of tuples of objects and methods to decorate.
        function_in (function, optional): Function applied on the input. Defaults to None.
        function_out (function, optional): Function applied on the output. Defaults to None.
    """
    for obj, method in functions:
        decorated = art_decorate_single_func(input_decorator, output_decorator)(
            getattr(obj, method)
        )
        setattr(obj, method, decorated)


class BatchSaver:
    """Save images from batch to debug_images folder"""

    def __init__(self, how_many_batches=10, image_key_name=INPUT):
        """
        Args:
            how_many_batches (int, optional): How many batches to save. Defaults to 10.
            image_key_name (str, optional): under what . Defaults to "input".
        """
        self.time = 0
        self.how_many_batches = how_many_batches
        self.image_key_name = image_key_name

        self.img_path = Path("debug_images")
        self.img_path.mkdir(exist_ok=True, parents=True)

    def __call__(self, data: Dict):
        """
        Args:
            data (Dict): Dictionary that was passed to some model function
        """
        if self.time < self.how_many_batches:
            img_ = data[self.image_key_name]
            min_, max_ = img_.min(), img_.max()
            img_ = ((img_ - min_) / (max_ - min_)) * 255
            save_image(img_, self.img_path / f"{self.time}.png")
        self.time += 1


class LogInputStats:
    """Log input stats to art logger"""

    def __init__(self, suppress_stdout=True, custom_logger=None):
        """
        Args:
            suppress_stdout (bool, optional): Whether to suppress stdout. Defaults to True.
            custom_logger (_type_, optional): By default art_logger will be used. You can pass your custom logger if you want. Defaults to None.
        """
        import lovely_tensors as lt

        lt.monkey_patch()
        self.logger = art_logger if custom_logger is None else custom_logger

        if suppress_stdout and custom_logger is None:
            self.logger = supress_stdout(self.logger)

    def __call__(self, *args, **kwargs):
        self.logger.info(f"Input stats: {args}, {kwargs}")


class EvolutionSaver:
    """Track evolution of logits for a given class"""

    def __init__(self, wanted_class_id: int):
        """
        Args:
            wanted_class_id (int): Which class to track
        """
        self.wanted_class_id = wanted_class_id
        self.logits = []
        self.time = 0

    def __call__(self, data: Dict):
        """Given dictionary with neural network output, save logits for a given class

        Args:
            data (Dict): Output of most likely predict step
        """
        targets = data[TARGET] == self.wanted_class_id
        logits = data[PREDICTION]

        wanted_logits = logits[targets].mean(dim=0)
        for i, logit in enumerate(wanted_logits):
            self.logits.append({"time": self.time, "logit": logit.item(), "class": i})

        self.time += 1

    def visualize(self):
        """Visualizes evolution of logits for a given class."""
        import pandas as pd

        if len(self.logits) == 0:
            print("Step was not run and logits are empty")
            return

        df = pd.DataFrame(self.logits)
        df = df.pivot(index="time", columns="class", values="logit")
        self.logits = []
        self.time = 0
        return df.plot(
            xlabel="epoch number",
            ylabel="logit value",
            title="Evolution of logits values when digit 1 is a target",
        )
