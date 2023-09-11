"""Module for atmsopheric functions."""
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import scipy.constants
from metpy.units import masked_array
from numpy import ma

import mwrpy.constants as con
from mwrpy import utils

HPA_TO_P = 100


def spec_heat(T: np.ndarray) -> np.ndarray:
    """Specific heat for evaporation (J/kg)"""
    return con.LATENT_HEAT - 2420.0 * (T - con.T0)


def vap_pres(q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Water vapor pressure (Pa)"""
    return q * con.RW * T


def t_dew_rh(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Dew point temperature (K) from relative humidity ()"""
    return (
        mpcalc.dewpoint_from_relative_humidity(
            masked_array(T, data_units="K"), masked_array(rh, data_units="")
        ).magnitude
        + con.T0
    )


def pot_tem(T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Potential temperature (K)"""
    p_baro = calc_p_baro(T, q, p, z)
    return mpcalc.potential_temperature(
        masked_array(p_baro, data_units="Pa"), masked_array(T, data_units="K")
    ).magnitude


def eq_pot_tem(
    T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Equivalent potential temperature (K)"""
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
    """Relative humidity ()"""
    return vap_pres(q, T) / calc_saturation_vapor_pressure(T)


def rh_err(T: np.ndarray, q: np.ndarray, dT: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Calculates relative humidity error propagation
    from absolute humidity and temperature ()
    """
    es = calc_saturation_vapor_pressure(T)
    drh_dq = con.RW * T / es
    des_dT = es * 17.67 * 243.5 / ((T - con.T0) + 243.5) ** 2
    drh_dT = q * con.RW / es**2 * (es - T * des_dT)
    drh = np.sqrt((drh_dq * dq) ** 2 + (drh_dT * dT) ** 2)

    return drh


def abs_hum(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    "Absolute humidity (kg/m^3)"
    es = calc_saturation_vapor_pressure(T)
    return (rh * es) / (con.RW * T)


def calc_p_baro(
    T: np.ndarray, q: np.ndarray, p: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Calculate pressure (Pa) in each level using barometric height formula"""
    Tv = mpcalc.virtual_temperature(
        masked_array(T, data_units="K"), masked_array(q, data_units="")
    ).magnitude
    p_baro = ma.masked_all(T.shape)
    p_baro[
        (~ma.getmaskarray(q).any(axis=1)) & (~ma.getmaskarray(T).any(axis=1)), 0
    ] = p[(~ma.getmaskarray(q).any(axis=1)) & (~ma.getmaskarray(T).any(axis=1))]
    for ialt in np.arange(len(z) - 1) + 1:
        p_baro[:, ialt] = p_baro[:, ialt - 1] * np.exp(
            -scipy.constants.g
            * (z[ialt] - z[ialt - 1])
            / (con.RS * np.mean([Tv[:, ialt], Tv[:, ialt - 1]]))
        )

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


def dir_avg(time: np.ndarray, spd: np.ndarray, drc: np.ndarray, win: float = 0.5):
    """Computes average wind direction (DEG) for a certain window length"""
    width = int(ma.round(win / ma.median(ma.diff(ma.masked_invalid(time)))))
    if (width % 2) != 0:
        width = width + 1
    seq = range(len(time))
    avg_dir = []
    for i in range(len(seq) - width + 1):
        avg_dir.append(winddir(spd[seq[i : i + width]], drc[seq[i : i + width]]))
    return np.array(avg_dir), width


def winddir(spd: np.ndarray, drc: np.ndarray):
    """Computes mean wind direction (deg)"""
    ve = -np.mean(spd * np.sin(np.deg2rad(drc)))
    vn = -np.mean(spd * np.cos(np.deg2rad(drc)))
    vdir = np.rad2deg(np.arctan2(ve, vn))
    if vdir < 180.0:
        Dv = vdir + 180.0
    elif vdir > 180.0:
        Dv = vdir - 180
    else:
        Dv = vdir
    return Dv


def find_lwcl_free(lev1: dict) -> tuple[np.ndarray, np.ndarray]:
    """Identifying liquid water cloud free periods using 31.4 GHz TB variability + IRT.
    Uses pre-defined time index and additionally returns status of IRT availability"""

    index = np.ones(len(lev1["time"]), dtype=np.float32) * np.nan
    status = np.ones(len(lev1["time"]), dtype=np.int32)
    freq_31 = np.where(np.round(lev1["frequency"][:], 1) == 31.4)[0]
    if len(freq_31) == 1:
        tb = np.squeeze(lev1["tb"][:, freq_31])
        tb[
            (lev1["pointing_flag"][:] == 1) | (lev1["elevation_angle"][:] < 89.0)
        ] = np.nan
        ind = utils.time_to_datetime_index(lev1["time"][:])
        tb_df = pd.DataFrame({"Tb": tb}, index=ind)
        tb_std = tb_df.rolling(
            pd.tseries.frequencies.to_offset("2min"), center=True, min_periods=10
        ).std()
        tb_mx = tb_std.rolling(
            pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
        ).max()

        if "irt" in lev1:
            tb_thres = 0.15
            irt = lev1["irt"][:, :]
            irt[irt == -999.0] = np.nan
            irt = np.nanmean(irt, axis=1) if irt.shape[1] > 1 else np.squeeze(irt)
            irt[
                (lev1["pointing_flag"][:] == 1) | (lev1["elevation_angle"][:] < 89.0)
            ] = np.nan
            irt_df = pd.DataFrame({"Irt": irt[:]}, index=ind)
            irt_mx = irt_df.rolling(
                pd.tseries.frequencies.to_offset("20min"), center=True, min_periods=100
            ).max()
            index[(irt_mx["Irt"] > 263.15) & (tb_mx["Tb"] > tb_thres)] = 1.0
            status[:] = 0

        tb_thres = 0.2
        index[(tb_mx["Tb"] > tb_thres)] = 1.0
        df = pd.DataFrame({"index": index}, index=ind)
        df = df.bfill(limit=120)
        df = df.ffill(limit=120)
        index = np.array(df["index"])
        index[(tb_mx["Tb"] < tb_thres) & (index != 1.0)] = 0.0
        index[(lev1["elevation_angle"][:] < 89.0) & (index != 1.0)] = 2.0

    return np.nan_to_num(index, nan=2).astype(int), status
