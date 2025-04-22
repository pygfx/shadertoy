"""Global configuration for pytest"""

import os
import sys

import numpy as np
import pytest

# Enable importing testutils from all test scripts
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "tests")))


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-screenshots",
        action="store_true",
        dest="regenerate_screenshots",
        default=False,
    )


@pytest.fixture(autouse=True)
def predictable_random_numbers():
    """
    Called at start of each test, guarantees that calls to random produce the same output over subsequent tests runs,
    see http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.seed.html
    """
    np.random.seed(0)


@pytest.fixture(autouse=True, scope="session")
def numerical_exceptions():
    """
    Ensure any numerical errors raise a warning in our test suite
    The point is that we enforce such cases to be handled explicitly in our code
    Preferably using local `with np.errstate(...)` constructs
    """
    np.seterr(all="raise")


@pytest.fixture(autouse=True, scope="function")
def force_offscreen():
    os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["RENDERCANVAS_FORCE_OFFSCREEN"]
