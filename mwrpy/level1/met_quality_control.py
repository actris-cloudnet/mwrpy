"""Module for meteorological sensor quality control"""
import numpy as np

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
        if name in data:
            ind = np.where(
                (data[name][:] < params["met_thresholds"][bit][0])
                | (data[name][:] > params["met_thresholds"][bit][1])
            )
            data["met_quality_flag"][ind] = setbit(data["met_quality_flag"][ind], bit)
