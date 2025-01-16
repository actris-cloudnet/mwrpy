"""Module for LWP offset correction."""

from itertools import groupby

import numpy as np
import pandas as pd

from mwrpy import utils


def correct_lwp_offset(
    lev1: dict,
    lwp_org: np.ndarray,
    index: np.ndarray,
    qf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """This function corrects Lwp offset using the
    2min Lwp standard deviation and the water vapor
    channel as proxy for a humidity dependent threshold.

    Args:
        lev1: Level 1 data.
        lwp_org: Lwp array.
        index: Index to use.
        qf: Quality flag
    """
    lwcl_i = lev1["liquid_cloud_flag"][:][index]
    ind = utils.time_to_datetime_index(lev1["time"][:][index])
    lwp = np.copy(lwp_org)
    lwp[(lev1["elevation_angle"][index] < 89.0) | (qf > 0)] = np.nan
    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    offset = "3min" if np.nanmean(np.diff(lev1["time"])) < 1.8 else "10min"
    lwp_std = lwp_df.rolling(
        pd.tseries.frequencies.to_offset(offset), center=True, min_periods=50
    ).std()
    offset = "20min" if np.nanmean(np.diff(lev1["time"])) < 1.8 else "60min"
    lwp_max = lwp_std.rolling(
        pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
    ).max()
    lwp_min = lwp_df.rolling(
        pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
    ).min()
    freq_31 = np.where(np.round(lev1["frequency"][:].data, 1) == 31.4)[0]
    freq_22 = np.where(np.round(lev1["frequency"][:].data, 1) == 22.2)[0]
    tb = lev1["tb"][:][index, :]
    tb[lev1["elevation_angle"][index] < 89.0] = np.nan
    tb_rat = pd.DataFrame(
        {"Tb": np.squeeze(tb[:, freq_22] / tb[:, freq_31])},
        index=ind,
    )
    tb_max = tb_rat.rolling(
        pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
    ).max()

    lwp[
        (
            (lwcl_i != 0)
            & (lwp_max["Lwp"] > (tb_max["Tb"] * 0.0025 / 1.5))
            & (lwp_min["Lwp"] > -0.025)
        )
        | (lwp > 0.06)
        | (lwp_max["Lwp"] > (tb_max["Tb"] * 0.0025))
    ] = np.nan
    if not np.isnan(lwp).all():
        lwp[
            (np.abs(lwp - np.nanmedian(lwp)) > 0.005) & (lwp_min["Lwp"].values > -0.025)
        ] = np.nan

    seqs_all = [(key, len(list(val))) for key, val in groupby(np.isfinite(lwp))]
    seqs = np.array(
        [
            (key, sum(s[1] for s in seqs_all[:i]), length)
            for i, (key, length) in enumerate(seqs_all)
            if bool(key) is True
        ]
    )
    if len(seqs) > 0:
        for sec in range(len(seqs)):
            if (
                lev1["time"][seqs[sec, 1] + seqs[sec, 2] - 1]
                - lev1["time"][seqs[sec, 1]]
                < 180
            ):
                lwp[seqs[sec, 1] : seqs[sec, 1] + seqs[sec, 2] - 1] = np.nan

    lwp_df = pd.DataFrame({"Lwp": lwp}, index=ind)
    lwp_offset = lwp_df.rolling(
        pd.tseries.frequencies.to_offset("60min"), center=True, min_periods=10
    ).mean()
    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.bfill()

    lwp_offset.loc[(np.isnan(lwp_offset["Lwp"])) | (lwp_org == -999.0), "Lwp"] = 0.0
    lwp_org -= lwp_offset["Lwp"].values
    return lwp_org, lwp_offset["Lwp"].values
