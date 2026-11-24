"""Explicit native-library settings for reproducible case execution."""
from contextlib import contextmanager, nullcontext


@contextmanager
def numerical_runtime(config):
    """Apply an optional case thread profile and restore the caller's settings.

    Native linear-algebra reduction order can affect an ill-conditioned fitted
    surrogate. Thread count is therefore recorded as an execution parameter.
    This context does not change any scientific threshold, seed or objective.
    """
    from threadpoolctl import threadpool_info, threadpool_limits

    threads = config.get("numerical_threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
        raise ValueError("numerical_threads must be a positive integer")
    scope = threadpool_limits(limits=threads) if threads is not None else nullcontext()
    with scope:
        fields = ("user_api", "internal_api", "num_threads", "prefix", "version", "threading_layer", "architecture")
        yield {
            "requested_threads": threads,
            "policy": "explicit_case_profile" if threads is not None else "inherited_runtime",
            "native_libraries": [{key: library[key] for key in fields if key in library} for library in threadpool_info()],
        }
