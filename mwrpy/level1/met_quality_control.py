"""Module for meteorological sensor quality control."""

import metpy.calc as mpcalc
import numpy as np
from metpy.units import masked_array

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
            altitude = masked_array(params["altitude"], data_units="m")
            pressure = mpcalc.height_to_pressure_std(altitude).to("pascal").magnitude
            threshold_low = pressure - 10000
            threshold_high = pressure + 10000
        else:
            threshold_low, threshold_high = params["met_thresholds"][bit]
        ind = (data[name][:] < threshold_low) | (data[name][:] > threshold_high)
        data["met_quality_flag"][ind] = setbit(data["met_quality_flag"][ind], bit)
