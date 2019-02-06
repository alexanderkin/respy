""" This module provides the fixtures for the PYTEST runs.
"""
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    """ Each test is executed with the same random seed.
    """
    np.random.seed(1423)


@pytest.fixture(scope="function", autouse=True)
def fresh_directory():
    """ Each test is executed in a fresh directory.
    """
    os.chdir(tempfile.mkdtemp())
