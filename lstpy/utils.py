import numpy as np
from .load import load_xr


def load_histogram(filename, n_bins=1, max_values=2**13, chunk='auto',
                   remove_rare_ch=None):
    return histogram(load_xr(filename, chunk, join='inner',
                             remove_rare_ch=remove_rare_ch),
                     n_bins=n_bins, max_values=max_values)


def histogram(dataarray, n_bins=1, max_values=None):
    """
    Make a histogram from dataarray that are read from lstpy.load

    Parameters
    ----------
    dataarray: xr.Dataarray
        with dimension ['ch', 'events']
    n_bin: interger
        number of binning.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError('xarray is required for histogram')

    dataarray = dataarray.transpose('events', 'ch')
    if max_values is not None:
        if not hasattr(max_values, '__len__'):
            max_values = (max_values, ) * len(dataarray['ch'])
        # remove event if at least one channel is out of range
        idx = (dataarray < np.array(max_values)).prod(
            'ch').astype(bool)
        dataarray = dataarray.isel(events=idx)
    else:
        max_values = dataarray.max('events') + 1
    max_values = np.array(max_values)

    if n_bins != 1:
        if not hasattr(n_bins, '__len__'):
            n_bins = (n_bins, ) * len(dataarray['ch'])
        dataarray = dataarray // np.array(n_bins)
    else:
        n_bins = (1, ) * len(dataarray['ch'])
    n_bins = np.array(n_bins)
    bins = (np.floor(max_values / n_bins)).astype(int)
    max_values = (np.ceil(max_values / n_bins)).astype(int)

    data, edges = np.histogramdd(dataarray.values, bins=bins,
                                 range=[(0, m) for m in max_values])

    # construct coordinate and dimensions
    coords = {}
    dims = []
    for i, ch in enumerate(dataarray['ch']):
        k = 'ADC{}'.format(ch.values.item())
        dims.append(k)
        coords[k] = 0.5 * (edges[i][1:] + edges[i][:-1]) * n_bins[i]
    return xr.DataArray(data, dims=dims, coords=coords)
