"""Module for meteorological sensor quality control."""

import atmoslib
import numpy as np
from numpy import ma

from mwrpy.utils import setbit


def apply_met_qc(data: dict, params: dict, altitude: float | None) -> None:
    """This function performs quality control of meteorological sensor data.

    Args:
        data: Level 1 data.
        params: Site specific parameters.
        altitude: Altitude of the site in meters.

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
            alt = params.get("altitude", altitude)
            alt = ma.masked if alt is None else alt
            pressure = atmoslib.isa_pressure(alt)
            threshold_low = pressure - 10000
            threshold_high = pressure + 10000
        else:
            threshold_low, threshold_high = params["met_thresholds"][bit]
        ind = (data[name][:] < threshold_low) | (data[name][:] > threshold_high)
        data["met_quality_flag"][ind] = setbit(data["met_quality_flag"][ind], bit)
        if name == "rainfall_rate" and np.all(data[name][:] * 1000.0 * 3600.0 > 60.0):
            # Remove rain flag for unreliable rainfall rates.
            data["quality_flag"] &= ~(1 << 5)
            data["met_quality_flag"][:] = setbit(data["met_quality_flag"][:], bit)
        if (
            name == "rainfall_rate"
            and len(((data[name] == 0.0) & (data["rain"] == 1)).nonzero()[0]) > 0
            and len(((data[name] > 0.0) & (data["rain"] == 1)).nonzero()[0])
            / len(((data[name] == 0.0) & (data["rain"] == 1)).nonzero()[0])
            < 10
        ):
            # Rely on rain sensor if flag is set too often with zero rainfall rate.
            # Mark times with flag set and zero rainfall rate as unreliable.
            data["quality_flag"] &= ~(1 << 5)
            ind_rain = data[name] > 0.0
            data["quality_flag"][ind_rain] = setbit(data["quality_flag"][ind_rain], 5)
            no_rain = (data[name] == 0.0) & (data["rain"] == 1)
            data["met_quality_flag"][no_rain] = setbit(
                data["met_quality_flag"][no_rain], bit
            )
