"""Module for meteorological sensor quality control."""

import numpy as np

import mwrpy.constants as con
from mwrpy.utils import setbit


def apply_met_qc(data: dict, params: dict) -> None:
    """This function performs quality control of meteorological sensor data.

    Args:
        data: Level 1 data.
        params: Site specific parameters.

    Returns:
        None

    Raises:
        RuntimeError:

    Example:
        from level1.met_quality_control import apply_met_qc
        apply_met_qc('lev1_data', 'params')

    """
    data["met_quality_flag"] = np.zeros(len(data["time"]), dtype=np.int32)
    var_name = [
        "air_temperature",
        "relative_humidity",
        "air_pressure",
        "rainfall_rate",
        "wind_direction",
        "wind_speed",
    ]

    for bit, name in enumerate(var_name):
        if name not in data:
            continue
        if name == "air_pressure":
            gamma = 6.5 / 1000.0
            pressure = con.p0 * (1 - (gamma / 288.0) * params["altitude"]) ** (
                con.g0 / (gamma * con.RS)
            )
            threshold_low = pressure - 10000
            threshold_high = pressure + 10000
        else:
            threshold_low, threshold_high = params["met_thresholds"][bit]
        ind = (data[name][:] < threshold_low) | (data[name][:] > threshold_high)
        data["met_quality_flag"][ind] = setbit(data["met_quality_flag"][ind], bit)
        if name == "rainfall_rate" and np.all(data[name][:] * 1000.0 * 3600.0 > 60.0):
            data["met_quality_flag"][:] = setbit(data["met_quality_flag"][:], bit)
