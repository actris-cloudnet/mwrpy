"""Module for atmospheric helper functions."""

from os import PathLike

import numpy as np
import pandas as pd

from mwrpy import utils
from mwrpy.level1 import droplet_mwrpy


def dir_avg(
    time: np.ndarray, spd: np.ndarray, drc: np.ndarray, win: int = 30
) -> np.ndarray:
    """Computes average wind direction (DEG) for a certain window length."""
    ve = spd * np.sin(np.deg2rad(drc))
    vn = spd * np.cos(np.deg2rad(drc))
    ind = utils.time_to_datetime_index(time)
    components = pd.DataFrame({"ve": ve, "vn": vn}, index=ind)

    avg_comp = components.rolling(
        pd.offsets.Minute(win), center=True, min_periods=1
    ).mean()
    avg_dir = np.rad2deg(np.arctan2(-avg_comp["ve"], -avg_comp["vn"])).values

    return np.where(avg_dir < 180.0, avg_dir + 180.0, avg_dir - 180.0)


def find_lwcl_free(
    lev1: dict, path_to_lidar: str | PathLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """Identifying liquid water cloud free periods using 31.4 GHz TB variability.
    Uses water vapor channel as proxy for a humidity dependent threshold.
    """
    index = np.ones(len(lev1["time"]), dtype=np.int32)
    status = np.zeros(len(lev1["time"]), dtype=np.int32)

    # Different frequencies for window and water vapor channels depending on instrument type
    freq_win = np.where(
        (np.isclose(np.round(lev1["frequency"][:], 1), 31.4))
        | (np.isclose(np.round(lev1["frequency"][:], 1), 190.8))
    )[0]
    freq_win = np.array([freq_win[0]]) if len(freq_win) > 1 else freq_win
    freq_wv = np.where(
        (np.isclose(np.round(lev1["frequency"][:], 1), 22.2))
        | (np.isclose(np.round(lev1["frequency"][:], 1), 183.9))
    )[0]
    if len(freq_win) == 1 and len(freq_wv) == 1:
        tb = np.squeeze(lev1["tb"][:, freq_win])
        tb[(lev1["pointing_flag"][:] == 1) | (lev1["elevation_angle"][:] < 89.0)] = (
            np.nan
        )
        ind = utils.time_to_datetime_index(lev1["time"][:])
        tb_df = pd.DataFrame({"Tb": tb}, index=ind)
        offset = "3min" if np.nanmean(np.diff(lev1["time"])) < 1.8 else "10min"
        tb_std = tb_df.rolling(
            pd.tseries.frequencies.to_offset(offset), center=True, min_periods=50
        ).std()
        offset = "20min" if np.nanmean(np.diff(lev1["time"])) < 1.8 else "60min"
        tb_mx = tb_std.rolling(
            pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
        ).max()

        tb_wv = np.squeeze(lev1["tb"][:, freq_wv])
        tb_rat = pd.DataFrame({"Tb": tb_wv / tb}, index=ind)
        tb_rat = tb_rat.rolling(
            pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
        ).max()

        index_rem = np.array(range(len(lev1["time"])))
        if path_to_lidar:
            # Use lidar data (Cloudnet format) to identify liquid water clouds
            lidar = utils.read_lidar(path_to_lidar)
            mwr_ind = [
                i
                for i, tt in enumerate(lev1["time"])
                if np.min(np.abs(tt - lidar["time"])) < 600
            ]
            lidar_ind = [
                i
                for i, tt in enumerate(lidar["time"])
                if np.min(np.abs(tt - lev1["time"])) < 600
            ]
            if len(mwr_ind) > 0:
                fact = (
                    0.75
                    if np.isclose(np.round(lev1["frequency"][freq_win], 1), 190.8)
                    else 0.1
                )
                liquid_from_lidar = droplet_mwrpy.find_liquid(
                    lidar,
                    lev1["time"][mwr_ind],
                    tb_mx["Tb"].iloc[mwr_ind].values,
                    tb_th=np.nanmedian(tb_rat["Tb"]) * fact,
                )
                liquid_flag = pd.DataFrame(
                    {"lf": liquid_from_lidar[lidar_ind]},
                    index=utils.time_to_datetime_index(lidar["time"][lidar_ind]),
                )
                liquid_flag = liquid_flag.resample(
                    "20min", origin="start", closed="left", label="left", offset="10min"
                ).max()
                liquid_flag = liquid_flag.reindex(
                    tb_df.index[mwr_ind], method="nearest"
                )
                liquid_flag = liquid_flag.fillna(value=2.0)
                index[mwr_ind] = np.array(liquid_flag["lf"][:].values, dtype=np.int32)
                status[mwr_ind] = 1
                index_rem = np.setxor1d(index_rem, mwr_ind)

        index[
            index_rem[
                tb_mx["Tb"].iloc[index_rem] < tb_rat["Tb"].iloc[index_rem] * 0.075
            ]
        ] = 0

        df = pd.DataFrame({"index": index}, index=ind)
        df = df.bfill(limit=120)
        df = df.ffill(limit=120)
        index = np.array(df["index"])
        index[(lev1["elevation_angle"][:] < 89.0) & (index != 0)] = 2

    return index, status
