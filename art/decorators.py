from typing import List, Tuple

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
                # first arguments is the `self` object. We don't want to pass it to the visualizing function
                to_be_passed = args[1:]
                visualizing_function_in(*to_be_passed, **kwargs)
            output = func(*args, **kwargs)
            if visualizing_function_out is not None:
                visualizing_function_out(output)
            return output

        return wrapper

    return decorator


def art_decorate(
    functions: List[Tuple[object, str]],
    function_in=None,
    function_out=None,
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
        decorated = art_decorate_single_func(function_in, function_out)(
            getattr(obj, method)
        )
        setattr(obj, method, decorated)
