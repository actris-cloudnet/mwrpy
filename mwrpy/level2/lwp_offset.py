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

    lwcl_i = lev1["liquid_cloud_flag"][index]
    ind = utils.time_to_datetime_index(lev1["time"][index])
    lwp_df = pd.DataFrame({"Lwp": lwp_org}, index=ind)
    lwp_std = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("2min"), center=True, min_periods=10
    ).std()
    lwp_max = lwp_std.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).max()
    lwp = np.copy(lwp_org)
    lwp[
        (lwcl_i != 0)
        | (lwp > 0.04)
        | (lev1["elevation_angle"][index] < 89.0)
        | (lwp_max["Lwp"][:] > 0.003)
    ] = np.nan
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_offset = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).mean()

    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.fillna(method="bfill")
    lwp_offset["Lwp"][np.isnan(lwp_offset["Lwp"])] = 0.0
    lwp_org -= lwp_offset["Lwp"].values

    return lwp_org, lwp_offset["Lwp"].values
