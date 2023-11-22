from typing import Any, Dict

def log(argument: Dict): ...
def save(configFile: str): ...

class Image:
    def __init__(self, image):
        self.image = image

run: Any
