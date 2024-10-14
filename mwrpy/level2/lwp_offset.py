"""Module for LWP offset correction."""

import numpy as np
import pandas as pd

from mwrpy import utils


def correct_lwp_offset(
    lev1: dict, lwp_org: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """This function corrects Lwp offset using the
    2min Lwp standard deviation and the water vapor
    channel as proxy for a humidity dependent threshold.

    Args:
        lev1: Level 1 data.
        lwp_org: Lwp array.
        index: Index to use.
    """
    if "elevation_angle" in lev1:
        elevation_angle = lev1["elevation_angle"][:]
    else:
        elevation_angle = 90 - lev1["zenith_angle"][:]

    lwcl_i = lev1["liquid_cloud_flag"][:][index]
    ind = utils.time_to_datetime_index(lev1["time"][:][index])
    lwp = np.copy(lwp_org)
    lwp[elevation_angle[index] < 89.0] = np.nan
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_std = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("2min"), center=True, min_periods=40
    ).std()
    lwp_max = lwp_std.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).max()
    freq_31 = np.where(np.round(lev1["frequency"][:].data, 1) == 31.4)[0]
    freq_22 = np.where(np.round(lev1["frequency"][:].data, 1) == 22.2)[0]
    tb = lev1["tb"][:][index, :]
    tb[elevation_angle[index] < 89.0] = np.nan
    tb_rat = pd.DataFrame(
        {"Tb": np.squeeze(tb[:, freq_22] / tb[:, freq_31])},
        index=ind,
    )
    tb_max = tb_rat.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).max()

    lwp[
        ((lwcl_i != 0) & (lwp_max["Lwp"] > (tb_max["Tb"] * 0.0025 / 1.5)))
        | (lwp > 0.06)
        | (lwp_max["Lwp"] > (tb_max["Tb"] * 0.0025))
    ] = np.nan
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_offset = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
    ).mean()
    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.bfill()

    lwp_mx = lwp_offset.rolling(
        pd.tseries.frequencies.to_offset("60min"), center=True, min_periods=100
    ).max()
    lwp_mn = lwp_offset.rolling(
        pd.tseries.frequencies.to_offset("60min"), center=True, min_periods=100
    ).min()
    lwp_offset.loc[
        (lwp_mx["Lwp"][:] - lwp_mn["Lwp"][:]) / tb_max["Tb"] > (tb_max["Tb"] * 0.002),
        "Lwp",
    ] = np.nan
    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.bfill()
    lwp_offset.loc[(np.isnan(lwp_offset["Lwp"])) | (lwp_org == -999.0), "Lwp"] = 0.0
    lwp_org -= lwp_offset["Lwp"].values
    return lwp_org, lwp_offset["Lwp"].values
