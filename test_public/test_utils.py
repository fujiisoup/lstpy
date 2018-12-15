import os

import numpy as np
import pytest
import lstpy


THIS_DIR = os.path.dirname(__file__)
file = THIS_DIR + '/data/Neon_KLL_002.lst'


@pytest.mark.parametrize(('n_bin', 'max_values'), [
    (1, 1000), (2, (1000, 1000)), (100, None)
])
def test_histogram(n_bin, max_values):
    hist = lstpy.load_histogram(file, n_bin, max_values)
