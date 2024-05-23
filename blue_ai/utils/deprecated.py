"""
Allow for marking certain functions and classes as deprecated, mainly to be
able to mark things that are preserved to not break scripts
"""

import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


# Examples


@deprecated
def some_old_function(x, y):
    return x + y


class SomeClass:
    @deprecated
    def some_old_method(self, x, y):
        return x + y
