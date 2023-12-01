"""Module for LWP offset correction"""
import numpy as np
import pandas as pd

from mwrpy import utils


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
    ind = utils.time_to_datetime_index(lev1["time"][index])
    lwp = np.copy(lwp_org)
    lwp[elevation_angle[index] < 89.0] = np.nan
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_std = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("2min"), center=True, min_periods=10
    ).std()
    lwp_max = lwp_std.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).max()
    lwp[(lwcl_i != 0) | (lwp > 0.06) | (lwp_max["Lwp"][:] > 0.003)] = np.nan
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_offset = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).mean()

    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.bfill()
    lwp_offset["Lwp"][(np.isnan(lwp_offset["Lwp"])) | (lwp_org == -999.0)] = 0.0
    lwp_org -= lwp_offset["Lwp"].values
    return lwp_org, lwp_offset["Lwp"].values
