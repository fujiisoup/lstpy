# lstpy

A small library to read output files MPA3 system lst file.
lstpy is full-python code but thanks to `numba` and multi-threading, the readout is fast enough.

# Install

```
python setup.py install
```

# Requirements

lstpy requires [numba](https://numba.pydata.org/numba-doc/dev/index.html)
and [numpy](https://docs.scipy.org/doc/).
Anaconda distribution might be the easiest way to install them.

```
conda install numba
conda install numpy
```

[xarray](https://xarray.pydata.org) is an optional dependency, but the use of
xarray greatly simplifies the further analysis.

# Usage

```python
import lstpy

header, (values, time, ch, events) = lstpy.load(filename)
```

or with xarray, then you can use

```python
>>> import lstpy

>>> lstpy.load_xr(filename)

<xarray.DataArray (ch: 2, events: 28991)>
array([[8191, 2679, 1431, ..., 2102, 1491, 1635],
       [ 973,  973,  973, ..., 4379, 4380, 4381]], dtype=uint16)
Coordinates:
  * ch       (ch) uint64 2 3
  * events   (events) uint64 1 2 3 4 5 6 ... 30141 30142 30143 30144
    time     (events) uint32 15 163 196 223 311 ... 409913 409940
Attributes:
    ctm:          400
    ...
```
