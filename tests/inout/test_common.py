"""Test the common methods for classes."""
import inspect

from turtlemd.inout.common import get_parameter_type, inspect_function

# Define some functions that we can inspect to get
# the types of arguments.


def foo1(arg1, arg2, *arg3, arg4=10, arg5=101, **arg6):
    pass


def foo2(a, b=10, *args, c, **kwargs):
    pass


def foo3():
    pass


def test_get_parameter_type():
    """Test that we can get the type of parameters."""
    params = inspect.signature(foo1).parameters.values()
    param_types = [get_parameter_type(i) for i in params]
    correct = ["args", "args", "varargs", "kwargs", "kwargs", "keywords"]
    assert param_types == correct


def test_inspect_function():
    """Test that we can inspect functions to get parameters."""
    params = inspect_function(foo2)
    assert params == {
        "args": ["a"],
        "kwargs": ["b", "c"],
        "varargs": ["args"],
        "keywords": ["kwargs"],
    }
    params = inspect_function(foo3)
    assert params == {}
