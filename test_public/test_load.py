import os

import numpy as np
import pytest
import lstpy


THIS_DIR = os.path.dirname(__file__)
file = THIS_DIR + '/data/Neon_KLL_002.lst'
# file2 = THIS_DIR + '/data/Biedermann036.lst'
file3 = THIS_DIR + '/data/binary_example_3dimenions_mpa4.lst'


@pytest.mark.parametrize(('filename', 'chunk', 'join'), [
    (file, None, 'inner'),
    (file3, None, 'inner'),
])
def test_load_xr(filename, chunk, join):
    data = lstpy.load_xr(filename, chunk=chunk, join=join)
    # make sure it can be saved as netcdf
    data.to_netcdf('test.nc')
    os.remove('test.nc')
    assert set(['time', 'ch', 'events']) == set(data.coords.keys())
    # make sure at least join='inner', it will be the uint16
    if join == 'inner':
        assert data.dtype == np.uint16
        assert data['ch'].dtype == np.int8
        assert data['time'].dtype == np.uint32

    assert len(data['ch']) > 0
    assert len(data['events']) > 0
    assert data.sum() > 0
    assert (data > 0).all()
