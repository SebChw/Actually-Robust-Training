from typing import List, Tuple

"""
Idea is as follows:
- Instead of thid visualize_img_on_input , we can give class with a state there.
- We don't need to restrict ourselves to just visualizations, we can store information about prediction dynamics, log mean values of tensors etc.
- State of this object is updated either by user on every call and/or by lightning module (set which epoch we at etc.)
"""


def visualize(visualizing_function_in=None, visualizing_function_out=None):
    """
    Decorator for visualizing input and output of a function.

    Args:
        visualizing_function_in (function, optional): Function to visualize input. Defaults to None.
        visualizing_function_out (function, optional): Function to visualize output. Defaults to None.

    Returns:
        function: Decorated function.
    """
    def decorator_visualize_input(func):
        """
        Decorator for visualizing input of a function.

        Args:
            func (function): Function to decorate.
        """
        def wrapper_visualize_input(*args, **kwargs):
            """
            Wrapper for visualizing input of a function.

            Returns:
                function: Decorated function.
            """
            if visualizing_function_in is not None:
                to_be_passed = args[1:]
                visualizing_function_in(*to_be_passed, **kwargs)
            output = func(*args, **kwargs)
            if visualizing_function_out is not None:
                visualizing_function_out(output)
            return output

        return wrapper_visualize_input

    return decorator_visualize_input


def set_visualization(
    functions: List[Tuple[object, str]],
    visualizing_function_in=None,
    visualizing_function_out=None,
):
    """
    Set visualization for a list of functions.

    Args:
        functions (List[Tuple[object, str]]): List of tuples of objects and methods to decorate.
        visualizing_function_in (function, optional): Function to visualize input. Defaults to None.
        visualizing_function_out (function, optional): Function to visualize output. Defaults to None.
    """
    for obj, method in functions:
        decorated = visualize(visualizing_function_in, visualizing_function_out)(
            getattr(obj, method)
        )
        setattr(obj, method, decorated)
