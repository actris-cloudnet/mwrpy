"""Module for LWP offset correction"""
import numpy as np
import pandas as pd

from mwrpy import utils
from mwrpy.atmos import find_lwcl_free


def correct_lwp_offset(
    lev1: dict, lwp_org: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """This function corrects Lwp offset using the
    2min standard deviation of the 31.4 GHz channel and IR temperature

    Args:
        lev1: Level 1 data.
        lwp_org: Lwp array.
        index: Index to use.
    """

    if "elevation_angle" in lev1:
        elevation_angle = lev1["elevation_angle"][:]
    else:
        elevation_angle = 90 - lev1["zenith_angle"][:]

    lwcl_i = lev1["liquid_cloud_flag"][index]
    lwp = np.copy(lwp_org)
    lwp[(lwcl_i != 0) | (lwp > 0.04) | (elevation_angle[index] < 89.0)] = np.nan
    ind = utils.time_to_datetime_index(lev1["time"][index])
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_std = lwp_df.rolling("2min", center=True, min_periods=10).std()
    lwp_max = lwp_std.rolling("20min", center=True, min_periods=100).max()
    lwp_df[lwp_max > 0.002] = np.nan
    lwp_offset = lwp_df.rolling("20min", center=True, min_periods=100).mean()

    if np.isnan(lwp_offset["Lwp"][-1]):
        lwp_offset["Lwp"][-1] = 0.0
    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.fillna(method="bfill")
    lwp_offset["Lwp"][np.isnan(lwp_offset["Lwp"])] = 0.0
    lwp_org -= lwp_offset["Lwp"].values

    return lwp_org, lwp_offset["Lwp"].values
