from typing import Any, Mapping, Union

class StringifyValue:
    def __init__(self, value: Any):
        self.__value = value
    @property
    def value(self):
        return self.__value
    def __str__(self):
        return str(self.__value)
    def __repr__(self):
        return repr(self.__value)

def stringify_unsupported(value: Any) -> Union[StringifyValue, Mapping]: ...
