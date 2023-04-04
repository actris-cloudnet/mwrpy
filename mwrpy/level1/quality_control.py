"""Quality control for level1 data."""
import datetime

import ephem
import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy import ma

from mwrpy.utils import get_coeff_list, setbit

Fill_Value_Float = -999.0
Fill_Value_Int = -99


def apply_qc(site: str, data: dict, params: dict) -> None:
    """This function performs the quality control of level 1 data.
    Args:
        site: Name of site.
        data: Level 1 data.
        params: Site specific parameters.

    Returns:
        None

    Raises:
        RuntimeError:

    Example:
        from level1.quality_control import apply_qc
        apply_qc('site', 'lev1_data', 'params')

    """

    data["quality_flag"] = np.zeros(data["tb"].shape, dtype=np.int32)
    data["quality_flag_status"] = np.zeros(data["tb"].shape, dtype=np.int32)

    # if params["flag_status"][3] == 0:
    #     c_list = get_coeff_list(site, "tbx")
    c_list = get_coeff_list(site, "tbx")
    if len(c_list) > 0:
        ind_bit4, _ = spectral_consistency(data, c_list)
    else:
        params["flag_status"][3] = 1
    ind_bit6 = np.where(data["rain"] == 1)
    ind_bit7 = orbpos(data, params)

    for freq, _ in enumerate(data["frequency"]):
        # Bit 1: Missing TB-value
        if params["flag_status"][0] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 0
            )
        else:
            ind = np.where(data["tb"][:, freq] == Fill_Value_Float)
            data["quality_flag"][ind, freq] = setbit(data["quality_flag"][ind, freq], 0)

        # Bit 2: TB threshold (lower range)
        if params["flag_status"][1] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 1
            )
        else:
            ind = np.where(data["tb"][:, freq] < params["TB_threshold"][0])
            data["quality_flag"][ind, freq] = setbit(data["quality_flag"][ind, freq], 1)

        # Bit 3: TB threshold (upper range)
        if params["flag_status"][2] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 2
            )
        else:
            ind = np.where(data["tb"][:, freq] > params["TB_threshold"][1])
            data["quality_flag"][ind, freq] = setbit(data["quality_flag"][ind, freq], 2)

        # Bit 4: Spectral consistency threshold
        if params["flag_status"][3] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 3
            )
        else:
            ind = np.where(ind_bit4[:, freq] == 1)
            data["quality_flag"][ind, freq] = setbit(data["quality_flag"][ind, freq], 3)

        # Bit 5: Receiver sanity
        if params["flag_status"][4] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 4
            )
        else:
            ind = np.where(data["status"][:, freq] == 1)
            data["quality_flag"][ind, freq] = setbit(data["quality_flag"][ind, freq], 4)

        # Bit 6: Rain flag
        if params["flag_status"][5] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 5
            )
        else:
            data["quality_flag"][ind_bit6, freq] = setbit(
                data["quality_flag"][ind_bit6, freq], 5
            )

        # Bit 7: Solar/Lunar flag
        if params["flag_status"][6] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 6
            )
        else:
            data["quality_flag"][ind_bit7, freq] = setbit(
                data["quality_flag"][ind_bit7, freq], 6
            )

        # Bit 8: TB offset threshold
        if params["flag_status"][7] == 1:
            data["quality_flag_status"][:, freq] = setbit(
                data["quality_flag_status"][:, freq], 7
            )
        # else:


def orbpos(data: dict, params: dict) -> np.ndarray:
    """Calculates sun & moon elevation/azimuth angles
    and returns index for observations in the direction of the sun"""

    sun = {}
    sun["azimuth_angle"] = np.zeros(data["time"].shape) * Fill_Value_Float
    sun["elevation_angle"] = np.zeros(data["time"].shape) * Fill_Value_Float
    moon = {}
    moon["azimuth_angle"] = np.zeros(data["time"].shape) * Fill_Value_Float
    moon["elevation_angle"] = np.zeros(data["time"].shape) * Fill_Value_Float

    sol = ephem.Sun()
    lun = ephem.Moon()
    obs_loc = ephem.Observer()

    for ind, time in enumerate(data["time"]):
        obs_loc.lat, obs_loc.lon = str(data["latitude"][ind]), str(
            data["longitude"][ind]
        )
        obs_loc.elevation = data["altitude"][ind]
        obs_loc.date = datetime.datetime.utcfromtimestamp(time).strftime(
            "%Y/%m/%d %H:%M:%S"
        )
        sol.compute(obs_loc)
        sun["elevation_angle"][ind] = np.rad2deg(sol.alt)
        sun["azimuth_angle"][ind] = np.rad2deg(sol.az)
        lun.compute(obs_loc)
        moon["elevation_angle"][ind] = np.rad2deg(lun.alt)
        moon["azimuth_angle"][ind] = np.rad2deg(lun.az)

    sun["rise"], moon["rise"] = data["time"][0], data["time"][0]
    sun["set"], moon["set"] = (
        data["time"][0] + 24.0 * 3600.0,
        data["time"][0] + 24.0 * 3600.0,
    )
    i_sun = np.where(sun["elevation_angle"] > 0.0)[0]
    if len(i_sun) > 0:
        sun["rise"] = data["time"][i_sun[0]]
        sun["set"] = data["time"][i_sun[-1]]
    i_moon = np.where(moon["elevation_angle"] > 0.0)[0]
    if len(i_moon) > 0:
        moon["rise"] = data["time"][i_moon[0]]
        moon["set"] = data["time"][i_moon[-1]]

    flag_ind = np.where(
        (
            (data["elevation_angle"] != Fill_Value_Float)
            & (data["elevation_angle"] <= np.max(sun["elevation_angle"]) + 10.0)
            & (data["time"] >= sun["rise"])
            & (data["time"] <= sun["set"])
            & (data["elevation_angle"] >= sun["elevation_angle"] - params["saf"])
            & (data["elevation_angle"] <= sun["elevation_angle"] + params["saf"])
            & (data["azimuth_angle"] >= sun["azimuth_angle"] - params["saf"])
            & (data["azimuth_angle"] <= sun["azimuth_angle"] + params["saf"])
        )
        | (
            (data["elevation_angle"] <= np.max(moon["elevation_angle"]) + 10.0)
            & (data["time"] >= moon["rise"])
            & (data["time"] <= moon["set"])
            & (data["elevation_angle"] >= moon["elevation_angle"] - params["saf"])
            & (data["elevation_angle"] <= moon["elevation_angle"] + params["saf"])
            & (data["azimuth_angle"] >= moon["azimuth_angle"] - params["saf"])
            & (data["azimuth_angle"] <= moon["azimuth_angle"] + params["saf"])
        )
    )

    return flag_ind


def spectral_consistency(data: dict, c_file: list) -> np.ndarray:
    """Applies spectral consistency coefficients for given frequency index
    and returns indices to be flagged"""

    flag_ind = np.zeros(data["tb"].shape, dtype=np.int32)
    abs_diff = ma.masked_all(data["tb"].shape, dtype=np.float32)
    tb_ret = np.ones(data["tb"].shape) * np.nan

    for ifreq, _ in enumerate(data["frequency"]):
        with nc.Dataset(c_file[ifreq]) as coeff:
            _, freq_ind, coeff_ind = np.intersect1d(
                data["frequency"],
                coeff["freq"],
                assume_unique=False,
                return_indices=True,
            )
            ele_ind = np.where(
                (data["elevation_angle"][:] > coeff["elevation_predictand"][:] - 0.6)
                & (data["elevation_angle"][:] < coeff["elevation_predictand"][:] + 0.6)
                & (data["pointing_flag"][:] == 0)
            )[0]

            if (ele_ind.size > 0) & (freq_ind.size > 0):
                tb_ret[ele_ind, ifreq] = (
                    coeff["offset_mvr"][:]
                    + np.sum(
                        coeff["coefficient_mvr"][coeff_ind].T
                        * np.array(data["tb"])[np.ix_(ele_ind, freq_ind)],
                        axis=1,
                    )
                    + np.sum(
                        coeff["coefficient_mvr"][
                            coeff_ind + (len(data["frequency"]) - 1)
                        ].T
                        * np.array(data["tb"])[np.ix_(ele_ind, freq_ind)] ** 2,
                        axis=1,
                    )
                )

                tb_df = pd.DataFrame(
                    {"Tb": (data["tb"][:, ifreq] - tb_ret[:, ifreq])},
                    index=pd.to_datetime(data["time"][:], unit="s"),
                )
                tb_mean = tb_df.resample(
                    "20min", origin="start", closed="left", label="left", offset="10min"
                ).mean()
                tb_mean = tb_mean.reindex(tb_df.index, method="nearest")

                fact = [2.5, 3.5]  # factor for receiver retrieval uncertainty
                # flag for individual channels based on channel retrieval uncertainty
                flag_ind[
                    ele_ind[
                        (
                            np.abs(
                                tb_df["Tb"].values[ele_ind]
                                - tb_mean["Tb"].values[ele_ind]
                            )
                            > coeff["predictand_err"][0]
                            * fact[data["receiver"][ifreq] - 1]
                        )
                    ],
                    ifreq,
                ] = 1
                abs_diff[:, ifreq] = ma.masked_invalid(
                    np.abs(data["tb"][:, ifreq] - tb_ret[:, ifreq])
                )

    th_rec = [1.0, 2.0]  # threshold for receiver mean absolute difference
    # receiver flag based on mean absolute difference
    for _, rec in enumerate(data["receiver_nb"]):
        flag_ind[
            np.ix_(
                ma.mean(abs_diff[:, data["receiver"] == rec], axis=1) > th_rec[rec - 1],
                data["receiver"] == rec,
            )
        ] = 1

    return flag_ind, tb_ret
