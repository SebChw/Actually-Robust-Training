from typing import List, Tuple

"""
Idea is as follows:
- Instead of thid visualize_img_on_input , we can give class with a state there.
- We don't need to restrict ourselves to just visualizations, we can store information about prediction dynamics, log mean values of tensors etc.
- State of this object is updated either by user on every call and/or by lightning module (set which epoch we at etc.)
"""


def visualize(visualizing_function_in=None, visualizing_function_out=None):
    def decorator_visualize_input(func):
        def wrapper_visualize_input(*args, **kwargs):
            if visualizing_function_in is not None:
                visualizing_function_in(*args, **kwargs)
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
    for obj, method in functions:
        decorated = visualize(visualizing_function_in, visualizing_function_out)(
            getattr(obj, method)
        )
        setattr(obj, method, decorated)

        if hasattr(obj, "reset_pipelines"):
            obj.reset_pipelines()


if __name__ == "__main__":
    """Just to test how this works."""

    def visualize_img_on_input(X):
        print("Visualizing input")

    def visualize_img_on_output(X):
        print("Visualizing output")

    class Model:
        def forward(self, X):
            print("Running foward")
            return X

    my_net = Model()
    set_visualization(
        [(my_net, "forward")], visualize_img_on_input, visualize_img_on_output
    )
    my_net.forward(1)
