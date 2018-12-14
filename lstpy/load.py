from os.path import getsize
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial

import numpy as np
from numba import njit


def load(filename, chunk=None):
    """
    Load list file.

    Parameters
    ----------
    filename: path to the file
    chunk: integer or None
        If integer, minibutch readout is performed.

    Returns
    -------
    values: 1d np.ndarray of uint16 size n
        Data values
    time: 1d np.ndarray of uint32 size n
        Time in ms for each entries.
    ch: 1d np.ndarray of uint8 size n
        Which analog-digital convertor (ch) does each entry of data belong to.
    """
    filesize = getsize(filename)
    with open(filename, 'rb') as file:
        file, header, version, is_ascii = _read_header(file)
        if version == 3:
            return header, _load_np(file, filesize, chunk)
        elif version == 4:
            return header, _load_np4(file, filesize, chunk, is_ascii)


def _parse_header(header):
    header_dict = OrderedDict()
    prefix = ''
    for line in header:
        # TODO temporary skip this
        if line[0] == ';':
            continue

        line = line.strip()
        if '[' in line and ']' in line:
            prefix = line[1: line.find(']')] + '.'
            if len(prefix) + 2 == len(line):
                continue
            line = line[line.find(']') + 1:].strip()

        kv = line.split('=')
        if len(kv) == 2:
            k, v = kv
        else:
            k, v = 'other', line
        k = prefix + k

        try:
            header_dict[k] = int(v)
        except ValueError:
            try:
                header_dict[k] = float(v)
            except ValueError:
                header_dict[k] = v

    return header_dict


def load_xr(filename, chunk=None, join='inner', remove_rare_ch=0.001):
    """
    Load list file and return as xr.DataArray

    Parameters
    ----------
    filename: path to the file
    chunk: integer or None
        If integer, minibutch readout is performed.
    join: 'inner' or 'outer'
        How to handle the missing values.
    remove_rare_ch: float
        remove the rare channels that are very rare

    Returns
    -------
    result: xr.DataArray
        with dimensions of ('event', 'ch')

    Note
    ----
    An example result is something like

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
    """
    import xarray as xr

    header, (values, time, ch, events) = load(filename, chunk)

    dataarray =  xr.DataArray(values, dims=['entry'],
                              coords={'time': ('entry', time),
                                      'ch': ('entry', ch),
                                      'events': ('entry', events)},
                              attrs=_parse_header(header))

    channels, counts = np.unique(ch, return_counts=True)
    valids = [True] * len(channels)
    if remove_rare_ch is not None:
        for i in range(len(channels)):
            if counts[i] / len(ch) < remove_rare_ch:
                valids[i] = False
    channels = channels[valids]
    channels = channels[channels >= 0]

    dataarrays = []
    for ch in channels:
        da = dataarray[dataarray['ch'] == ch]
        da = da.swap_dims({'entry': 'events'})
        da['ch'] = da['ch'][0].values
        dataarrays.append(da)

    return xr.concat(xr.align(*dataarrays, join=join),
                     dim=xr.DataArray(channels, dims='ch', name='ch'))


def _read_header(file):
    '''Process the listfile header.'''
    header = []
    while True:
        line = file.readline().decode('utf-8').strip()
        if line in ('[LISTDATA]', '[DATA]'):
            break
        header.append(line)

    is_ascii = False
    if '[MPA3]' in header[0]:
        version = 3
    elif '[MPA4A]' in header[0]:
        version = 4
        is_ascii = True
    else:
        raise NotImplementedError('{} is not supported.'.format(header[0]))

    return file, header, version, is_ascii


def _load_np(file, filesize, chunk):
    pos = file.tell()
    size = (filesize - pos) // np.dtype('u2').itemsize
    data = np.memmap(file, dtype='<u2', mode='r', offset=pos,
                     shape=(size, ))
    n = len(data)
    if chunk is None:
        # read into memory
        return decode(data.copy())[:4]  # do not return t_last

    # find a separation of each blocks
    pos = 0
    slices = []
    while pos < len(data) - 1:
        assert data[pos + 1] == 0x4000
        num_lines = get_sync_pos(data[pos: pos + chunk])
        if num_lines == 0:
            num_lines = len(data) - pos
        if num_lines == 0:
            break
        slices.append(slice(pos, pos + num_lines))
        pos += num_lines

    pool = Pool()
    results = pool.map(_decode_copy, [data[sl] for sl in slices])

    values = []
    time = []
    ch = []
    events = []
    t_last_cumsum = 0
    events_last_cumsum = 0
    for res in results:
        v, t, c, e, tl = res
        values.append(v)
        time.append(t + t_last_cumsum)
        t_last_cumsum += tl
        ch.append(c)
        events.append(e + events_last_cumsum)
        if len(e) > 0:
            events_last_cumsum += e[-1]

    return (np.concatenate(values), np.concatenate(time),
            np.concatenate(ch), np.concatenate(events))


_bitmask = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                     0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000,
                     0x8000], dtype='<u2')
OxFF = np.array([0xFFFF], dtype='<u2')


@njit
def get_sync_pos(data):
    # find a position of the last timer
    for j in range(len(data), 0, -1):
        if (data[j-2: j] == OxFF).all() and data[j-3] == 0x4000:
            return j - 4  # the last sync position
    return 0


# function to run in parallel
def _decode_copy(data):
    # read a part of data into memory
    return decode(data.copy())


@njit
def decode(data):
    n = len(data)

    values = np.zeros(n, dtype=np.uint16)
    ch = np.zeros(n, dtype=np.int8)
    time = np.zeros(n, dtype=np.uint32)
    events = np.zeros(n, dtype=np.uint32)

    l = 0  # index for data values
    t_count = 0  # index for the timing
    j = 0
    event_id = 0
    timer_event = True

    while j < n - 1:
        da = data[j: j+2].reshape(2)[::-1]
        if da[0] == 0x4000:  # timer
            t_count += 1
            alive_adcs = da[1]
            timer_event = True
            j += 2

        elif (da == OxFF).all():  # sync
            timer_event = False
            j += 2

        elif timer_event:
            # possible defect signal, since event signal can only follow
            # sync signal or event signal.
            # skip a file until the next sync event
            j = _next_timer_pos(data, j)
        else:  # possible event
            highest_byte = ((da[0] >> 8) & 0xFF)
            bit30 = (highest_byte & _bitmask[6]) != 0

            if bit30:  # invalid data
                j = _next_timer_pos(data, j)
            else:
                event_id += 1
                bit31 = (highest_byte & _bitmask[7]) != 0
                bit28 = (highest_byte & _bitmask[4]) != 0
                active_adcs = da[1]
                if bit28:
                    # there is still no way to find whether bit28 is RTC or
                    # it is just a corruped data
                    j = _next_timer_pos(data, j)
                    continue
                    # raise NotImplementedError

                # skip 1 byte if bit31 is high
                j = (j + 3) if bit31 else (j + 2)

                for i, bmask in enumerate(_bitmask):
                    if (active_adcs & bmask) != 0:
                        values[l] = data[j]
                        if (alive_adcs & bmask) == 0:
                            ch[l] = -1  # set -1 for dead channels
                        else:
                            ch[l] = i
                        time[l] = t_count
                        events[l] = event_id
                        l += 1
                        j += 1

    return values[:l], time[:l], ch[:l], events[:l], t_count


@njit
def _next_timer_pos(data, j):
    n = len(data)
    for i in range(j + 2, n - 2):
        if (data[i: i + 2] == OxFF).all() and data[i - 1] == 0x4000:
            return i - 2
    return n


def _load_np4(file, filesize, chunk, is_ascii):
    """
    For MPA4 format
    """
    pos = file.tell()
    size = (filesize - pos) // np.dtype('u2').itemsize
    if is_ascii:
        str2int = partial(int, base=16)
        # TODO enable chunk
        data = np.loadtxt(file, converters={0: str2int}, dtype=np.int64)
    else:
        raise NotImplementedError(
            'binary format for MPA4 system is not yet Implemented.')

    n = len(data)
    if chunk is None:
        # read into memory
        return decode4(data.copy())[:4]  # do not return t_last
    else:
        raise NotImplementedError('chunk is not supported for MPA4 data.')


@njit
def decode4(data):
    n = len(data)

    values = np.zeros(n, dtype=np.uint16)
    ch = np.zeros(n, dtype=np.int8)
    time = np.zeros(n, dtype=np.uint32)
    events = np.zeros(n, dtype=np.uint32)

    l = 0  # index for data values
    t_count = 0  # index for the timing
    j = 0
    event_id = 0
    timer_event = True

    adc_bmask = np.array([0b00000001, 0b00000010, 0b00000100, 0b00001000,
                          0b00010000, 0b00100000, 0b01000000, 0b10000000])

    while j < n - 1:
        da = data[j]
        if da & 0b1000 == 8:  # timer
            t_count += 1
            alive_adcs = ~((da & 0xFF00) >> 8)
            j += 1

        elif (da & 0x40) >> 6 == 0:  # single ADC
            event_id += 1
            time[l] = (da >> 32) & 0xFFFFFFFF
            values[l] = (da >> 16) & 0xFFFF
            j += 1
            l += 1

        elif (da & 0x40) >> 6 == 1:  # coincidence ADC
            event_id += 1
            time[l] = t_count
            aux1 = (da & 0b100) >> 3
            aux2 = (da & 0b1000) >> 4

            active_adcs = (da & 0xFF00) >> 8
            pos = 1
            for i, bmask in enumerate(adc_bmask):
                if (active_adcs & bmask) != 0:
                    if (alive_adcs & bmask) == 0:
                        ch[l] = -1  # set -1 for dead channels
                    else:
                        ch[l] = i
                    time[l] = t_count
                    events[l] = event_id
                    values[l] = (data[j] >> (pos * 16)) & 0xFFFF
                    l += 1
                    pos += 1
                    if pos == 4:
                        pos = 0
                        j += 1

            if pos != 0:
                j += 1
        else:
            raise ValueError  # TODO skip to the next timer event

    return values[:l], time[:l], ch[:l], events[:l], t_count
