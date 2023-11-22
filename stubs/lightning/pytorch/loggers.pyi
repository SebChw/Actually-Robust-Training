from typing import Any, List, Union

class Logger:
    experiment: Any
    run: Any
    def add_tags(self, tags: Union[List[str], str]):
        """
        Adds tags to the Neptune experiment.

        Args:
            tags (Union[List[str], str]): Tags to add.
        """
        pass

class WandbLogger(Logger): ...
class NeptuneLogger(Logger): ...
