"""Test the common methods for classes."""
import inspect
import logging

import pytest

from turtlemd.inout.common import (
    get_args_and_kwargs,
    get_parameter_type,
    initiate_instance,
    inspect_function,
)

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


class ClassNoArg:
    def __init__(self):
        self.a = 123


class ClassArg:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class ClassArgKwarg:
    def __init__(self, a, b=5):
        self.a = a
        self.b = b


class ClassKwarg:
    def __init__(self, a=10, b=5, c=101):
        self.a = a
        self.b = b
        self.c = c


class ClassVarArgs:
    def __init__(self, a, *args):
        self.a = a
        self.args = args


class ClassKeyword:
    def __init__(self, a, **kwargs):
        self.a = a
        self.kwargs = kwargs


def test_get_args_and_kwargs(caplog):
    """Test that we can get the args and kwargs of constructors."""
    # No arguments:
    settings = {}
    args, kwargs = get_args_and_kwargs(ClassNoArg, settings)
    assert not args
    assert not kwargs
    # Only positional arguments:
    settings = {"a": 1, "b": 2}
    args, kwargs = get_args_and_kwargs(ClassArg, settings)
    assert args == [1, 2]
    assert not kwargs
    # Only keyword:
    settings = {"a": 1, "c": 2}
    args, kwargs = get_args_and_kwargs(ClassKwarg, settings)
    assert not args
    assert kwargs == {"a": 1, "c": 2}
    # Positional and keyword:
    settings = {"a": 1, "b": 2}
    args, kwargs = get_args_and_kwargs(ClassArgKwarg, settings)
    assert args == [1]
    assert kwargs == {"b": 2}
    # Varargs:
    with pytest.raises(ValueError):
        get_args_and_kwargs(ClassVarArgs, settings)
    # Keywords:
    with pytest.raises(ValueError):
        get_args_and_kwargs(ClassKeyword, settings)
    # Missing arguments:
    with pytest.raises(ValueError):
        settings = {"a": 10}
        get_args_and_kwargs(ClassArg, settings)
    # Extra arguments:
    settings = {"a": 10, "b": 10, "__extra__": 101, "__stop__": 102}
    with caplog.at_level(logging.WARNING):
        get_args_and_kwargs(ClassArg, settings)
        assert "Ignored extra argument" in caplog.text
        assert "__extra__" in caplog.text
        assert "__stop__" in caplog.text


def test_initiate_instance():
    """Test that we can initiate classes."""
    settings = {}
    instance = initiate_instance(ClassNoArg, settings)
    assert isinstance(instance, ClassNoArg)
    assert instance.a == 123
    settings = {"a": -1, "b": -2}
    instance = initiate_instance(ClassArg, settings)
    assert isinstance(instance, ClassArg)
    assert instance.a == -1
    assert instance.b == -2
    settings = {"a": -1}
    instance = initiate_instance(ClassArgKwarg, settings)
    assert isinstance(instance, ClassArgKwarg)
    assert instance.a == -1
    assert instance.b == 5
    settings = {"a": -1, "b": -5}
    instance = initiate_instance(ClassArgKwarg, settings)
    assert instance.b == -5
    settings = {"b": -5}
    with pytest.raises(ValueError):
        initiate_instance(ClassArgKwarg, settings)
