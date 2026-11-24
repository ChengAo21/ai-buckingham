"""Native thread profiles are explicit, observed and scoped to one analysis."""
import pytest
import numpy as np
from threadpoolctl import threadpool_info

from ai_buckingham.execution import numerical_runtime
from ai_buckingham.cli import load_case


def counts():
    return [(library["prefix"], library["num_threads"]) for library in threadpool_info()]


def test_case_thread_profile_is_observed_and_restored_after_failure():
    np.eye(2) @ np.eye(2)
    original = counts()
    with pytest.raises(RuntimeError):
        with numerical_runtime({"numerical_threads": 2}) as runtime:
            assert runtime["requested_threads"] == 2
            assert runtime["native_libraries"]
            assert all(library["num_threads"] == 2 for library in runtime["native_libraries"])
            assert all("filepath" not in library for library in runtime["native_libraries"])
            raise RuntimeError("test scope restoration")
    assert counts() == original


def test_unconfigured_custom_analysis_inherits_native_runtime():
    original = counts()
    with numerical_runtime({}) as runtime:
        assert runtime["policy"] == "inherited_runtime"
        assert runtime["requested_threads"] is None
        assert counts() == original
    assert counts() == original


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, "20"])
def test_invalid_thread_profiles_are_rejected(value):
    with pytest.raises(ValueError):
        with numerical_runtime({"numerical_threads": value}):
            pass


def test_rough_pipe_has_the_verified_reference_thread_profile():
    assert load_case("rough_pipe")["numerical_threads"] == 20
