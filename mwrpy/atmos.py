"""Module for atmsopheric functions."""

from os import PathLike

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import scipy.constants
from metpy.units import masked_array
from numpy import ma

import mwrpy.constants as con
from mwrpy import utils
from mwrpy.level1 import droplet_mwrpy

HPA_TO_P = 100


def spec_heat(T: np.ndarray) -> np.ndarray:
    """Specific heat for evaporation (J/kg)."""
    return con.LATENT_HEAT - 2420.0 * (T - con.T0)


def vap_pres(q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Water vapor pressure (Pa)."""
    return q * con.RW * T


def t_dew_rh(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Dew point temperature (K) from relative humidity ()."""
    return (
        mpcalc.dewpoint_from_relative_humidity(
            masked_array(T, data_units="K"), masked_array(rh, data_units="")
        ).magnitude
        + con.T0
    )


def pot_tem(T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Potential temperature (K)."""
    p_baro = calc_p_baro(T, q, p, z)
    return mpcalc.potential_temperature(
        masked_array(p_baro, data_units="Pa"), masked_array(T, data_units="K")
    ).magnitude


def eq_pot_tem(
    T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Equivalent potential temperature (K)."""
    e = vap_pres(q, T)
    p_baro = calc_p_baro(T, q, p, z)
    Theta = pot_tem(T, q, p, z)
    return (
        Theta
        + (spec_heat(T) * con.MW_RATIO * e / (p_baro - e) / con.SPECIFIC_HEAT)
        * Theta
        / T
    )


def rel_hum(T: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Relative humidity ()."""
    return vap_pres(q, T) / calc_saturation_vapor_pressure(T)


def rh_err(T: np.ndarray, q: np.ndarray, dT: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Calculates relative humidity error propagation
    from absolute humidity and temperature ().
    """
    es = calc_saturation_vapor_pressure(T)
    drh_dq = con.RW * T / es
    des_dT = es * 17.67 * 243.5 / ((T - con.T0) + 243.5) ** 2
    drh_dT = q * con.RW / es**2 * (es - T * des_dT)
    drh = np.sqrt((drh_dq * dq) ** 2 + (drh_dT * dT) ** 2)

    return drh


def abs_hum(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Absolute humidity (kg/m^3)."""
    es = calc_saturation_vapor_pressure(T)
    return (rh * es) / (con.RW * T)


def calc_p_baro(
    T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Calculate pressure (Pa) in each level using barometric height formula."""
    Tv = mpcalc.virtual_temperature(
        masked_array(T, data_units="K"), masked_array(q, data_units="")
    ).magnitude
    Tv_half = (Tv[:, :-1] + Tv[:, 1:]) / 2
    dz = np.diff(z)
    dp = ma.exp(-scipy.constants.g * dz / (con.RS * Tv_half))
    tmp = np.insert(dp, 0, p, axis=1)
    p_baro = np.cumprod(tmp, axis=1)
    return p_baro


def calc_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Goff-Gratch formula for saturation vapor pressure (Pa) over water adopted by WMO.

    Args:
        temperature: Temperature (K).

    Returns:
        Saturation vapor pressure (Pa).
    """
    ratio = con.T0 / temperature
    inv_ratio = ratio**-1
    return (
        10
        ** (
            10.79574 * (1 - ratio)
            - 5.028 * np.log10(inv_ratio)
            + 1.50475e-4 * (1 - (10 ** (-8.2969 * (inv_ratio - 1))))
            + 0.42873e-3 * (10 ** (4.76955 * (1 - ratio)) - 1)
            + 0.78614
        )
    ) * HPA_TO_P


def c2k(T: np.ndarray) -> np.ndarray:
    """Converts Celsius to Kelvins."""
    return ma.array(T) + con.T0


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
