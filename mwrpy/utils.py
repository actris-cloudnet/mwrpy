"""Module for general helper functions."""

import datetime
import glob
import logging
import os
import time
from collections.abc import Iterable, Iterator
from typing import Any, Literal, NamedTuple

import netCDF4
import numpy as np
import pandas as pd
import yaml
from numpy import ma
from scipy import signal
from scipy.interpolate import RectBivariateSpline
from yaml.loader import SafeLoader

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
Epoch = tuple[int, int, int]


class MetaData(NamedTuple):
    long_name: str
    units: str
    standard_name: str | None = None
    definition: str | None = None
    comment: str | None = None
    retrieval_type: str | None = None
    retrieval_elevation_angles: str | None = None
    retrieval_frequencies: str | None = None
    retrieval_auxiliary_input: str | None = None
    retrieval_description: str | None = None


def seconds2hours(time_in_seconds: np.ndarray) -> np.ndarray:
    """Converts seconds since some epoch to fraction hour.

    Args:
        time_in_seconds: 1-D array of seconds since some epoch that starts on midnight.

    Returns:
        Time as fraction hour.

    Notes:
        Excludes leap seconds.
    """
    seconds_since_midnight = np.mod(time_in_seconds, SECONDS_PER_DAY)
    fraction_hour = seconds_since_midnight / SECONDS_PER_HOUR
    if fraction_hour[-1] == 0:
        fraction_hour[-1] = 24
    return fraction_hour


def epoch2unix(epoch_time, time_ref, epoch: Epoch = (2001, 1, 1)):
    """Converts seconds since some epoch to Unix time in UTC.

    Args:
        epoch_time: 1-D array of seconds since the given epoch.
        time_ref: HATPRO time reference (1: UTC, 0: Local Time)
        epoch: Epoch of the input time. Default is (2001,1,1,0,0,0).

    Returns:
        ndarray: Unix time in seconds since (1970,1,1,0,0,0).

    """
    delta = (
        datetime.datetime(*epoch) - datetime.datetime(1970, 1, 1, 0, 0, 0)
    ).total_seconds()
    unix_time = epoch_time + int(delta)
    if time_ref == 0:
        for index, _ in enumerate(unix_time):
            unix_time[index] = time.mktime(
                datetime.datetime.fromtimestamp(
                    unix_time[index], datetime.timezone.utc
                ).timetuple()
            )
    return unix_time


def isscalar(array: Any) -> bool:
    """Tests if input is scalar.

    By "scalar" we mean that array has a single value.

    Examples:
        >>> isscalar(1)
            True
        >>> isscalar([1])
            True
        >>> isscalar(np.array(1))
            True
        >>> isscalar(np.array([1]))
            True
    """
    arr = ma.array(array)
    if not hasattr(arr, "__len__") or arr.shape == () or len(arr) == 1:
        return True
    return False


def isbit(array: np.ndarray, nth_bit: int) -> np.ndarray:
    """Tests if nth bit (0,1,2..) is set.

    Args:
        array: Integer array.
        nth_bit: Investigated bit.

    Returns:
        Boolean array denoting values where nth_bit is set.

    Raises:
        ValueError: negative bit as input.

    Examples:
        >>> isbit(np.array([4, 5]), 1)
            array([False, False])
        >>> isbit(np.array([4, 5]), 2)
            array([ True,  True])

    See Also:
        utils.setbit()
    """
    if nth_bit < 0:
        raise ValueError("Negative bit number")
    mask = 1 << nth_bit
    return array & mask > 0


def setbit(array: np.ndarray, nth_bit: int) -> np.ndarray:
    """Sets nth bit (0, 1, 2..) on number.

    Args:
        array: Integer array.
        nth_bit: Bit to be set.

    Returns:
        Integer where nth bit is set.

    Raises:
        ValueError: negative bit as input.

    Examples:
        >>> setbit(np.array([0, 1]), 1)
            array([2, 3])
        >>> setbit(np.array([0, 1]), 2)
            array([4, 5])

    See Also:
        utils.isbit()
    """
    if nth_bit < 0:
        raise ValueError("Negative bit number")
    mask = 1 << nth_bit
    array |= mask
    return array


def interpol_2d(
    x_in: np.ndarray,
    array: ma.MaskedArray,
    x_new: np.ndarray,
) -> ma.MaskedArray:
    """Interpolates 2-D data in one dimension.

    Args:
        x_in: 1-D array with shape (n,).
        array: 2-D input data with shape (n, m).
        x_new: 1-D target vector with shape (N,).

    Returns:
        array: Interpolated data with shape (N, m).

    Notes:
        0-values are masked in the returned array.
    """
    result = np.zeros((len(x_new), array.shape[1]))
    array_screened = ma.masked_invalid(array, copy=True)  # data may contain nan-values
    for ind, values in enumerate(array_screened.T):
        if ma.is_masked(array):
            mask = ~values.mask
            if ma.any(values[mask]):
                result[:, ind] = np.interp(x_new, x_in[mask], values[mask])
        else:
            result[:, ind] = np.interp(x_new, x_in, values)
    result[~np.isfinite(result)] = 0
    masked = ma.make_mask(result)
    return ma.array(result, mask=np.invert(masked))


def interpolate_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: ma.MaskedArray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> ma.MaskedArray:
    """Linear interpolation of gridded 2d data.

    Args:
        x: 1-D array.
        y: 1-D array.
        z: 2-D array at points (x, y).
        x_new: 1-D array.
        y_new: 1-D array.

    Returns:
        Interpolated data.

    Notes:
        Does not work with nans. Ignores mask of masked data. Does not extrapolate.

    """
    fun = RectBivariateSpline(x, y, z, kx=1, ky=1)
    return fun(x_new, y_new)


def add_interpol1d(
    data0: dict, data1: ma.MaskedArray, time1: np.ndarray, output_name: str
) -> None:
    """Adds interpolated 1d field to dict, supporting masked arrays.

    Args:
        data0: Output dict.
        data1: Input field to be added & interpolated. Supports masked arrays.
        time1: Time of input field.
        output_name: Name of output field.
    """
    interpolated_data: np.ndarray = np.array([])
    n_time = len(data0["time"])

    itr = data1.T if data1.ndim > 1 else [data1]

    for input_data in itr:
        valid_mask = ~ma.getmaskarray(input_data)
        if ~valid_mask.all():
            result = ma.masked_all(n_time)
        else:
            valid_data = input_data[valid_mask]
            valid_time = time1[valid_mask]
            interpolated_values = np.interp(data0["time"], valid_time, valid_data)
            interpolated_mask = (
                np.interp(data0["time"], valid_time, valid_mask.astype(float)) < 0.5
            )
            result = ma.masked_array(interpolated_values, mask=interpolated_mask)
        interpolated_data = (
            result
            if len(interpolated_data) == 0
            else ma.vstack((interpolated_data, result))
        )
    if data1.ndim > 1:
        interpolated_data = np.reshape(interpolated_data.T, (n_time, -1))

    data0[output_name] = interpolated_data


def seconds2date(time_in_seconds: float, epoch: Epoch = (1970, 1, 1)) -> list:
    """Converts seconds since some epoch to datetime (UTC).

    Args:
        time_in_seconds: Seconds since some epoch.
        epoch: Epoch, default is (1970, 1, 1) (UTC).

    Returns:
        [year, month, day, hours, minutes, seconds] formatted as '05' etc (UTC).
    """
    epoch_in_seconds = datetime.datetime.timestamp(
        datetime.datetime(*epoch, tzinfo=datetime.timezone.utc)
    )
    timestamp = time_in_seconds + epoch_in_seconds
    return (
        datetime.datetime.utcfromtimestamp(timestamp)
        .strftime("%Y %m %d %H %M %S")
        .split()
    )


def str_to_numeric(value: str) -> int | float:
    """Converts string to number (int or float)."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def add_time_bounds(time_arr: np.ndarray, int_time: int) -> np.ndarray:
    """Adds time bounds."""
    time_bounds = np.empty((len(time_arr), 2), dtype=np.int32)
    time_bounds[:, 0] = time_arr - int_time
    time_bounds[:, 1] = time_arr

    return time_bounds


def get_coeff_list(site: str | None, prefix: str, coeff_files: list | None) -> list:
    """Returns list of .nc coefficient file(s)."""
    if coeff_files is not None:
        c_list = []
        for file in coeff_files:
            if f"{prefix.lower()}_" in file.lower():
                logging.debug("Using coefficient file: " + file)
                c_list.append(file)
        return sorted(c_list)

    assert isinstance(site, str)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s_list = [
        glob.glob(
            dir_path + "/site_config/" + site + "/coefficients/" + prefix.lower() + "*"
        ),
        glob.glob(
            dir_path + "/site_config/" + site + "/coefficients/" + prefix.upper() + "*"
        ),
    ]
    c_list = [x for x in s_list if x]

    if len(c_list) > 0:
        return sorted(c_list[0])
    logging.warning(
        "No coefficient files for product "
        + prefix
        + " found in directory "
        + "/site_config/"
        + site
        + "/coefficients/"
    )
    return c_list


def get_file_list(path_to_files: str, extension: str):
    """Returns file list for specified path."""
    f_list = sorted(glob.glob(path_to_files + "/*." + extension))
    if len(f_list) == 0:
        f_list = sorted(glob.glob(path_to_files + "/*." + extension.lower()))
    if len(f_list) == 0:
        logging.warning(
            "No binary files with extension "
            + extension
            + " found in directory "
            + path_to_files
        )
    return f_list


def read_config(site: str | None, key: Literal["global_specs", "params"]) -> dict:
    data = _read_hatpro_config_yaml()[key]
    if site is not None:
        data.update(_read_site_config_yaml(site)[key])
    return data


def _read_hatpro_config_yaml() -> dict:
    dir_name = os.path.dirname(os.path.realpath(__file__))
    inst_file = os.path.join(dir_name, "site_config", "hatpro.yaml")
    with open(inst_file, "r", encoding="utf8") as f:
        return yaml.load(f, Loader=SafeLoader)


def _read_site_config_yaml(site: str) -> dict:
    dir_name = os.path.dirname(os.path.realpath(__file__))
    site_file = os.path.join(dir_name, "site_config", site, "config.yaml")
    if not os.path.isfile(site_file):
        raise NotImplementedError(f"Error: site config file {site_file} not found")
    with open(site_file, "r", encoding="utf8") as f:
        return yaml.load(f, Loader=SafeLoader)


def update_lev1_attributes(attributes: dict, data_type: str) -> None:
    """Removes attributes that are not needed for specified Level 1 data type."""
    if data_type == "1B01":
        att_del = ["ir_instrument", "met_instrument", "_accuracy"]
        key = " "
    elif data_type == "1B11":
        att_del = [
            "instrument_manufacturer",
            "instrument_model",
            "instrument_generation",
            "instrument_hw_id",
            "instrument_calibration",
            "receiver",
            "date_of",
            "instrument_history",
            "met",
            "air",
            "relative",
            "wind",
            "rain",
        ]
        key = "ir_"
    elif data_type == "1B21":
        att_del = [
            "instrument_manufacturer",
            "instrument_model",
            "instrument_generation",
            "instrument_hw_id",
            "instrument_calibration",
            "receiver",
            "date_of",
            "instrument_history",
            "ir_instrument",
            "ir_accuracy",
        ]
        key = "met"
        attributes["source"] = "In Situ"

    for name in list(attributes.keys()):
        if any(x in name for x in att_del) & (name[0:3] != key):
            del attributes[name]


def read_nc_field_name(nc_file: str, name: str) -> str:
    """Reads selected variable name from a netCDF file.

    Args:
        nc_file: netCDF file name.
        name: Variable to be read, e.g. 'temperature'.

    Returns:
        str
    """
    with netCDF4.Dataset(nc_file) as nc:
        long_name = nc.variables[name].getncattr("long_name")
    return long_name


def read_nc_fields(nc_file: str, name: str) -> np.ndarray:
    """Reads selected variables from a netCDF file.

    Args:
        nc_file: netCDF file name.
        name: Variable to be read, e.g. 'lwp'.

    Returns:
        np.ndarray
    """
    assert os.path.isfile(nc_file), f"File {nc_file} does not exist."
    with netCDF4.Dataset(nc_file) as nc:
        return nc.variables[name][:]


def append_data(data_in: dict, key: str, array: ma.MaskedArray) -> dict:
    """Appends data to a dictionary field (creates the field if not yet present).

    Args:
        data_in: Dictionary where data will be appended.
        key: Key of the field.
        array: Numpy array to be appended to data_in[key].
    """
    data = data_in.copy()
    if key not in data:
        if array.ndim == 1:
            data[key] = array[:]
        else:
            data[key] = array[:, :]
    else:
        data[key] = ma.concatenate((data[key], array))
    return data


def convolve2DFFT(slab, kernel, max_missing=0.1):
    """2D convolution using fft.
    <slab>: 2d array, with optional mask.
    <kernel>: 2d array, convolution kernel.
    <max_missing>: real, max tolerable percentage of missing within any
                   convolution window.
                   E.g. if <max_missing> is 0.5, when over 50% of values
                   within a given element are missing, the center will be
                   set as missing (<res>=0, <resmask>=1). If only 40% is
                   missing, center value will be computed using the remaining
                   60% data in the element.
                   NOTE that out-of-bound grids are counted as missing, this
                   is different from convolve2D(), where the number of valid
                   values at edges drops as the kernel approaches the edge.
    Return <result>: 2d convolution.
    """
    assert np.ndim(slab) == 2, "<slab> needs to be 2D."
    assert np.ndim(kernel) == 2, "<kernel> needs to be 2D."
    assert kernel.shape[0] <= slab.shape[0], "<kernel> size needs to <= <slab> size."
    assert kernel.shape[1] <= slab.shape[1], "<kernel> size needs to <= <slab> size."
    # --------------Get mask for missing--------------
    slab[slab == 0.0] = np.nan
    slabcount = 1 - np.isnan(slab)
    # this is to set np.nan to a float, this won't affect the result as
    # masked values are not used in convolution. Otherwise, nans will
    # affect convolution in the same way as scipy.signal.convolve()
    # and the result will contain nans.
    slab = np.where(slabcount == 1, slab, 0)
    kernelcount = np.where(kernel == 0, 0, 1)
    result = signal.fftconvolve(slab, kernel, mode="same")
    result_mask = signal.fftconvolve(slabcount, kernelcount, mode="same")
    valid_threshold = (1.0 - max_missing) * np.sum(kernelcount)
    result /= np.sum(kernel)
    result[(result_mask < valid_threshold)] = np.nan

    return result


def date_string_to_date(date_string: str) -> datetime.date:
    """Convert YYYY-MM-DD to Python date."""
    date_arr = [int(x) for x in date_string.split("-")]
    return datetime.date(*date_arr)


def get_time() -> str:
    """Returns current UTC-time."""
    return f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} +00:00"


def get_date_from_past(n: int, reference_date: str | None = None) -> str:
    """Return date N-days ago.

    Args:
        n: Number of days to skip (can be negative, when it means the future).
        reference_date: Date as "YYYY-MM-DD". Default is the current date.

    Returns:
        str: Date as "YYYY-MM-DD".
    """
    reference = reference_date or get_time().split()[0]
    the_date = date_string_to_date(reference) - datetime.timedelta(n)
    return str(the_date)


def get_processing_dates(args) -> tuple[str, str]:
    """Returns processing dates."""
    if args.date is not None:
        start_date = args.date
        stop_date = get_date_from_past(-1, start_date)
    else:
        start_date = args.start
        stop_date = args.stop
    start_date = str(date_string_to_date(start_date))
    stop_date = str(date_string_to_date(stop_date))
    return start_date, stop_date


def _get_filename(prod: str, date_in: datetime.date, site: str) -> str:
    global_attributes = read_config(site, "global_specs")
    params = read_config(site, "params")
    if np.char.isnumeric(prod[0]):
        level = prod[0]
    else:
        level = "2"
    data_out_dir = os.path.join(
        params["data_out"], f"level{level}", date_in.strftime("%Y/%m/%d")
    )
    wigos_id = global_attributes["wigos_station_id"]
    filename = f"MWR_{prod}_{wigos_id}_{date_in.strftime('%Y%m%d')}.nc"
    return os.path.join(data_out_dir, filename)


def isodate2date(date_str: str) -> datetime.date:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def date_range(
    start_date: datetime.date, end_date: datetime.date
) -> Iterator[datetime.date]:
    """Returns range between two dates (datetimes)."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def time_to_datetime_index(time_array: np.ndarray) -> pd.DatetimeIndex:
    time_units = "s" if max(time_array) > 25 else "h"
    return pd.to_datetime(time_array, unit=time_units)


def copy_variables(
    source: netCDF4.Dataset, target: netCDF4.Dataset, keys: Iterable[str]
) -> None:
    """Copies variables (and their attributes) from one file to another.

    Args:
        source: Source object.
        target: Target object.
        keys: Variable names to be copied.

    """
    for key in keys:
        if key in source.variables:
            fill_value = getattr(source.variables[key], "_FillValue", False)
            variable = source.variables[key]
            var_out = target.createVariable(
                key,
                variable.datatype,
                variable.dimensions,
                fill_value=fill_value,
            )
            var_out.setncatts(
                {
                    k: variable.getncattr(k)
                    for k in variable.ncattrs()
                    if k != "_FillValue"
                }
            )
            var_out[:] = variable[:]


def copy_global(
    source: netCDF4.Dataset, target: netCDF4.Dataset, attributes: Iterable[str]
) -> None:
    """Copies global attributes from one file to another.

    Args:
        source: Source object.
        target: Target object.
        attributes: List of attributes to be copied.

    """
    source_attributes = source.ncattrs()
    for attr in attributes:
        if attr in source_attributes:
            setattr(target, attr, source.getncattr(attr))
