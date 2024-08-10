"""Utilities that can be used for calling functions on a particular rank."""
from functools import wraps



def rank_zero_only(fn, default=None):
    """Wrap a function to call internal function only in rank zero.

    Function that can be used as a decorator to enable a function/method being called only on global rank 0.

    """

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn
