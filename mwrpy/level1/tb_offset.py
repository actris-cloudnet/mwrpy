import datetime
import time

import numpy as np

from mwrpy.utils import epoch2unix


def correct_tb_offset(
    site: str, data: dict, params: dict, global_attributes: dict, rpg_cal: dict
) -> None:
    """This function performs the TB offset correction of level 1 data.
    Args:
        site: Name of site.
        data: Level 1 data.
        params: Site specific parameters.
        global_attributes: Level 1 global attributes.
        rpg_cal: Calibration history.

    Returns:
        None

    Raises:
        RuntimeError:

    Example:
        from level1.tb_offset import correct_tb_offset
        correct_tb_offset('site', 'lev1_data', 'params')

    """

    if global_attributes["instrument_calibration_status"] == "calibrated":
        cal_s = int(
            time.mktime(
                datetime.datetime.strptime(
                    global_attributes["receiver1_date_of_last_absolute_calibration"],
                    "%Y%m%d",
                ).timetuple()
            )
        )
        cal_e = cal_nex(params, data["time"][-1], rpg_cal)

        # if cal_s < cal_e:


def cal_nex(params: dict, time1: int, rpg_cal: dict) -> int:
    """Load and add information from ABSCAL.HIS file"""

    cal_t = time1
    rec_cal = np.where(
        (rpg_cal.data["cal1_t"] == 1)
        & (rpg_cal.data["cal2_t"] == 1)
        & (epoch2unix(rpg_cal.data["t1"], rpg_cal.header["_time_ref"]) > time1)
    )[0]
    if len(rec_cal) > 0:
        cal_nex = epoch2unix(rpg_cal.data["t1"][rec_cal[0]], rpg_cal.header["_time_ref"])

    return cal_t
