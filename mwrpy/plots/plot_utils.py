"""Module containing additional helper functions for plotting."""

import locale
from datetime import datetime, timezone

import netCDF4
import numpy as np
import pandas as pd
from numpy import ma, ndarray

from mwrpy.utils import (
    convolve2DFFT,
    isbit,
    read_config,
    seconds2hours,
    time_to_datetime_index,
)


def _get_ret_flag(
    nc_file: str,
    time: np.ndarray,
    variable: str,
    bits: int = 0,
    instrument_type: str | None = None,
) -> ndarray:
    """Returns quality flag for frequencies used in retrieval."""
    file = netCDF4.Dataset(nc_file)
    quality_flag = file.variables[variable + "_quality_flag"]
    time_variable = (
        seconds2hours(file.variables["time"][:])
        if np.max(file.variables["time"]) > 24
        else file.variables["time"][:]
    )
    _, index, _ = np.intersect1d(
        time_variable, time, assume_unique=True, return_indices=True
    )
    quality_flag = quality_flag[index]
    flag = np.zeros(len(time), np.int32)
    site = _read_location(nc_file)
    params = read_config(site, instrument_type, "params")

    if params["flag_status"][3] == 0 and bits == 0:
        flag[isbit(quality_flag[:], 3) > 0] = 1
    else:
        flag[quality_flag[:] > 0] = 1
    return flag


def _get_freq_flag(data: ndarray, bits: ndarray) -> ndarray:
    """Returns array of flag values for each frequency."""
    flag = np.ones(data.shape) * np.nan
    for i, bit in enumerate(bits):
        flag[isbit(data, bit)] = i + 1
    flag[isbit(data, 0)] = 0
    return flag


def _get_bit_flag(data: ndarray, bits: ndarray) -> ndarray:
    """Returns array of flag values for each bit."""
    flag = np.ones((len(data), len(bits))) * np.nan
    for i, bit in enumerate(bits):
        flag[isbit(data, bit), i] = i
    return flag


def _get_unmasked_values(
    data: ma.MaskedArray, time: ndarray
) -> tuple[ndarray, ndarray]:
    """Returns unmasked time and data."""
    if ma.is_masked(data) is False:
        return data, time
    good_values = ~data.mask
    return data[good_values], time[good_values]


def _nan_time_gaps(time: ndarray, tgap: float = 5.0 / 60.0) -> ndarray:
    """Finds time gaps bigger than 5min (default) and inserts nan."""
    time_diff = ma.diff(ma.masked_invalid(time))
    gaps = np.where(time_diff > tgap)[0] + 1
    if len(gaps) > 0:
        time[gaps[0 : ma.min([len(time), gaps[-1]])]] = ma.masked
    return time


def _gap_array(time: ndarray, case_date, tgap: float = 5.0 / 60.0) -> ndarray:
    """Returns edges of time gaps bigger than 5min (default).
    End of gap for current day is current time.
    """
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
    dtnow = datetime.now(tz=timezone.utc)
    day_e = 24.0
    if dtnow.strftime("%d %b %Y") == case_date.strftime("%d %b %Y"):
        day_e = dtnow.hour + dtnow.minute / 60.0 + dtnow.second / 3600.0
        if day_e - time[-1] < 2.0:
            day_e = time[-1]
    time_diff = np.diff(time, prepend=0.0, append=day_e)
    gaps = np.where(time_diff > tgap)[0]
    gtim = np.zeros((len(gaps), 2), np.float32)
    if len(gaps) > 0:
        for i, ind in enumerate(gaps):
            if ind < len(time):
                gtim[i, :] = [time[ind - 1], time[ind]]
    return gtim


def _calculate_rolling_mean(time: ndarray, data: ndarray, win: float = 0.5) -> ndarray:
    """Returns rolling mean."""
    if data.ndim == 1:
        ind = time_to_datetime_index(time)
        df = pd.DataFrame({"data": data}, index=ind)
        rolling_mean = (
            df.rolling(
                pd.offsets.Minute(int(np.floor(win * 60))), center=True, min_periods=1
            )
            .mean()
            .data
        )
    else:
        if time[-1] - time[0] < win:
            return data
        width = ma.max(
            (2, int(ma.round(win / ma.median(ma.diff(ma.masked_invalid(time))))))
        )
        data = ma.filled(data, np.nan)
        if (width % 2) != 0:
            width = width + 1
        rolling_window = np.ones((1, width)) * np.blackman(width)
        rolling_mean = convolve2DFFT(data, rolling_window.T, max_missing=0.1)
    return rolling_mean


def _dir_avg(
    time: np.ndarray, spd: np.ndarray, drc: np.ndarray, win: int = 30
) -> np.ndarray:
    """Computes average wind direction (DEG) for a certain window length."""
    ve = spd * np.sin(np.deg2rad(drc))
    vn = spd * np.cos(np.deg2rad(drc))
    ind = time_to_datetime_index(time)
    components = pd.DataFrame({"ve": ve, "vn": vn}, index=ind)

    avg_comp = components.rolling(
        pd.offsets.Minute(win), center=True, min_periods=1
    ).mean()
    avg_dir = np.rad2deg(np.arctan2(-avg_comp["ve"], -avg_comp["vn"]))

    return np.where(avg_dir < 180.0, avg_dir + 180.0, avg_dir - 180.0)


def _read_location(nc_file: str) -> str:
    """Returns site name."""
    with netCDF4.Dataset(nc_file) as nc:
        site_name = nc.site_location if "site_location" in nc.ncattrs() else nc.location
    return site_name
