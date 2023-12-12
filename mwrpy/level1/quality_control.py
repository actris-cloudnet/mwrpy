"""Quality control for level1 data."""
import datetime

import ephem
import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy import ma

from mwrpy.level1.rpg_bin import RpgBin
from mwrpy.level2.get_ret_coeff import get_mvr_coeff
from mwrpy.level2.write_lev2_nc import retrieval_input
from mwrpy.utils import get_coeff_list, setbit

Fill_Value_Float = -999.0
Fill_Value_Int = -99


def apply_qc(
    site: str | None, data_in: RpgBin, params: dict, coeff_files: list | None
) -> None:
    """This function performs the quality control of level 1 data.
    Args:
        site: Name of site.
        data_in: Level 1 data.
        params: Site specific parameters.
        coeff_files: Retrieval coefficients.

    Returns:
        None

    Raises:
        RuntimeError:

    Example:
        from level1.quality_control import apply_qc
        apply_qc('site', 'lev1_data', 'params')

    """
    data = data_in.data

    data["quality_flag"] = np.zeros(data["tb"].shape, dtype=np.int32)
    data["quality_flag_status"] = np.zeros(data["tb"].shape, dtype=np.int32)

    if params["flag_status"][3] == 0:
        ind_bit4 = spectral_consistency(data, site, coeff_files)
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

    sun: dict = {
        "azimuth_angle": np.zeros(data["time"].shape) * Fill_Value_Float,
        "elevation_angle": np.zeros(data["time"].shape) * Fill_Value_Float,
    }
    moon: dict = {
        "azimuth_angle": np.zeros(data["time"].shape) * Fill_Value_Float,
        "elevation_angle": np.zeros(data["time"].shape) * Fill_Value_Float,
    }

    sol = ephem.Sun()
    lun = ephem.Moon()
    obs_loc = ephem.Observer()

    for ind, time in enumerate(data["time"]):
        obs_loc.lat, obs_loc.lon = (
            str(data["latitude"][ind]),
            str(data["longitude"][ind]),
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
    )[0]

    return flag_ind


def spectral_consistency(
    data: dict, site: str | None, coeff_files: list | None
) -> np.ndarray:
    """Applies spectral consistency coefficients for given frequency index,
    writes 2S02 product and returns indices to be flagged"""

    flag_ind = np.zeros(data["tb"].shape, dtype=np.int32)
    abs_diff = ma.masked_all(data["tb"].shape, dtype=np.float32)
    data["tb_spectrum"] = np.ones(data["tb"].shape) * np.nan

    c_list = get_coeff_list(site, "spc", coeff_files)

    if len(c_list) > 0:
        # pylint: disable=unbalanced-tuple-unpacking
        (
            coeff,
            input_scale,
            input_offset,
            output_scale,
            output_offset,
            weights1,
            weights2,
            factor,
        ) = get_mvr_coeff(site, "spc", data["frequency"][:], coeff_files)
        ret_in = retrieval_input(data, coeff)
        ele_ang = 90.0
        ele_coeff = np.where(coeff["AG"] == ele_ang)[0]
        ele_ind = np.where(
            (data["elevation_angle"][:] > ele_ang - 0.5)
            & (data["elevation_angle"][:] < ele_ang + 0.5)
            & (data["pointing_flag"][:] == 0)
        )[0]
        coeff_ind = np.searchsorted(coeff["AL"], data["frequency"])
        c_w1, c_w2, fac = (
            weights1(data["elevation_angle"][ele_ind]),
            weights2(data["elevation_angle"][ele_ind]),
            factor(data["elevation_angle"][ele_ind]),
        )
        in_sc, in_os = (
            input_scale(data["elevation_angle"][ele_ind]),
            input_offset(data["elevation_angle"][ele_ind]),
        )
        op_sc, op_os = (
            output_scale(data["elevation_angle"][ele_ind]),
            output_offset(data["elevation_angle"][ele_ind]),
        )

        ret_in[ele_ind, 1:] = (ret_in[ele_ind, 1:] - in_os) * in_sc
        hidden_layer = np.ones((len(ele_ind), c_w1.shape[2] + 1), np.float32)
        hidden_layer[:, 1:] = np.tanh(
            fac[:].reshape((len(ele_ind), 1))
            * np.einsum("ijk,ij->ik", c_w1, ret_in[ele_ind, :])
        )
        data["tb_spectrum"][ele_ind, :] = (
            np.tanh(
                fac[:].reshape((len(ele_ind), 1))
                * np.einsum("ijk,ik->ij", c_w2[:, coeff_ind, :], hidden_layer)
            )
            * op_sc[:, coeff_ind]
            + op_os[:, coeff_ind]
        )

        for ifreq, _ in enumerate(data["frequency"]):
            tb_df = pd.DataFrame(
                {"Tb": (data["tb"][:, ifreq] - data["tb_spectrum"][:, ifreq])},
                index=pd.to_datetime(data["time"][:], unit="s"),
            )
            tb_mean = tb_df.resample(
                "20min", origin="start", closed="left", label="left", offset="10min"
            ).mean()
            tb_mean = tb_mean.reindex(tb_df.index, method="nearest")

            fact = [5.0, 7.0]  # factor for receiver retrieval uncertainty
            # flag for individual channels based on channel retrieval uncertainty
            flag_ind[
                (
                    np.abs(tb_df["Tb"].values[:] - tb_mean["Tb"].values[:])
                    > coeff["RM"][coeff_ind[ifreq], ele_coeff]
                    * fact[data["receiver"][ifreq] - 1]
                ),
                ifreq,
            ] = 1
            abs_diff[:, ifreq] = np.abs(
                data["tb"][:, ifreq] - data["tb_spectrum"][:, ifreq]
            )

    else:
        c_list = get_coeff_list(site, "tbx", coeff_files)

        for ifreq, _ in enumerate(data["frequency"]):
            with nc.Dataset(c_list[ifreq]) as cfile:
                _, freq_ind, coeff_ind = np.intersect1d(
                    data["frequency"],
                    cfile["freq"],
                    assume_unique=False,
                    return_indices=True,
                )
                ele_ind = np.where(
                    (
                        data["elevation_angle"][:]
                        > cfile["elevation_predictand"][:] - 0.5
                    )
                    & (
                        data["elevation_angle"][:]
                        < cfile["elevation_predictand"][:] + 0.5
                    )
                    & (data["pointing_flag"][:] == 0)
                )[0]

                if (ele_ind.size > 0) & (freq_ind.size > 0):
                    data["tb_spectrum"][ele_ind, ifreq] = (
                        cfile["offset_mvr"][:]
                        + np.sum(
                            cfile["coefficient_mvr"][coeff_ind].T
                            * np.array(data["tb"])[np.ix_(ele_ind, freq_ind)],
                            axis=1,
                        )
                        + np.sum(
                            cfile["coefficient_mvr"][
                                coeff_ind + (len(data["frequency"]) - 1)
                            ].T
                            * np.array(data["tb"])[np.ix_(ele_ind, freq_ind)] ** 2,
                            axis=1,
                        )
                    )

                    tb_df = pd.DataFrame(
                        {"Tb": (data["tb"][:, ifreq] - data["tb_spectrum"][:, ifreq])},
                        index=pd.to_datetime(data["time"][:], unit="s"),
                    )
                    tb_mean = tb_df.resample(
                        "20min",
                        origin="start",
                        closed="left",
                        label="left",
                        offset="10min",
                    ).mean()
                    tb_mean = tb_mean.reindex(tb_df.index, method="nearest")

                    fact = [3.0, 4.0]  # factor for receiver retrieval uncertainty
                    min_err = [0.2, 0.4]  # minimum channel error per receiver
                    # flag for individual channels based on retrieval uncertainty
                    flag_ind[
                        ele_ind[
                            (
                                np.abs(
                                    tb_df["Tb"].values[ele_ind]
                                    - tb_mean["Tb"].values[ele_ind]
                                )
                                > np.max(
                                    (
                                        cfile["predictand_err"][0],
                                        min_err[data["receiver"][ifreq] - 1],
                                    )
                                )
                                * fact[data["receiver"][ifreq] - 1]
                            )
                        ],
                        ifreq,
                    ] = 1

            abs_diff[:, ifreq] = ma.masked_invalid(
                np.abs(data["tb"][:, ifreq] - data["tb_spectrum"][:, ifreq])
            )

    th_rec = [1.5, 2.0]  # threshold for receiver mean absolute difference
    # receiver flag based on mean absolute difference
    for _, rec in enumerate(data["receiver_nb"]):
        flag_ind[
            np.ix_(
                ma.mean(abs_diff[:, data["receiver"] == rec], axis=1) > th_rec[rec - 1],
                data["receiver"] == rec,
            )
        ] = 1

    return flag_ind
