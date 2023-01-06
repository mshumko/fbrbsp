from typing import List, Union
import dateutil.parser
from datetime import timedelta, datetime
import copy

import numpy as np

_time_type = Union[datetime, str]
_time_range_type = List[_time_type]


def validate_time_range(time_range: _time_range_type) -> List[datetime]:
    """
    Validates tries to parse the time_range into datetime objects.
    """
    if time_range is None:
        return None

    assert isinstance(
        time_range, (list, tuple, np.ndarray)
    ), "time_range must be a list, tuple, or np.ndarray."
    assert len(time_range) == 2, "time_range must be a list or a tuple with start and end times."

    time_range_parsed = []

    for t in time_range:
        if isinstance(t, str):
            time_range_parsed.append(dateutil.parser.parse(t))
        elif isinstance(t, (int, float)):
            raise ValueError(f'Unknown time format, {t}')
        else:  # assume it is already a datetime-like object.
            time_range_parsed.append(t)

    time_range_parsed.sort()
    return time_range_parsed


def get_filename_times(time_range: _time_range_type, dt='hours') -> List[datetime]:
    """
    Returns the dates and times within time_range and in dt steps. It returns times
    with the smaller components than dt set to zero, and increase by dt. This is useful
    to create a list of dates and times to load sequential files.

    time_range: list[datetime or str]
        A start and end time to calculate the file dates.
    dt: str
        The time difference between times. Can be 'days', 'hours', or 'minutes'.
    """
    time_range = validate_time_range(time_range)
    assert dt in ['days', 'hours', 'minutes'], "dt must be 'day', 'hour', or 'minute'."
    # First we need to appropriately zero time_range[0] so that the file that contains the
    # time_range[0] is first. For example, if dt='hour' and time_range[0] is not at the
    # top of the hour, we zero the smaller time components. So
    # time_range[0] = '2010-01-01T10:29:32.000000' will be converted to 
    # '2010-01-01T10:00:00.000000'.
    zero_time_chunks = {'microsecond': 0, 'second': 0}
    # We don't need an if-statement if dt == 'minute'
    if dt == 'hours':
        zero_time_chunks['minute'] = 0
    elif dt == 'days':
        zero_time_chunks['minute'] = 0
        zero_time_chunks['hour'] = 0
    current_time = copy.copy(time_range[0].replace(**zero_time_chunks))

    times = []
    # Not <= in while loop because we don't want to download the final time if time_range[1]
    # exactly matches the end file name (you don't want the 11th hour data if time_range[1] 
    # is 'YYY-MM-DDT11:00:00').
    while current_time < time_range[1]:
        times.append(current_time)
        current_time += timedelta(**{dt: 1})
    return times