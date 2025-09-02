"""Quality control for level1 data."""

import datetime
import logging
from collections.abc import Sequence
from os import PathLike

import netCDF4 as nc
import numpy as np
import pandas as pd
import suncalc
from numpy import ma

from mwrpy.exceptions import MissingCoefficientsError
from mwrpy.level1.rpg_bin import RpgBin
from mwrpy.level2.get_ret_coeff import get_mvr_coeff
from mwrpy.level2.write_lev2_nc import retrieval_input
from mwrpy.utils import get_coeff_list, setbit


def apply_qc(
    site: str | None,
    data_in: RpgBin,
    params: dict,
    coeff_files: Sequence[str | PathLike] | None,
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

    # Bit 1: Missing TB-value
    if params["flag_status"][0] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 0)
    else:
        ind = ma.getmaskarray(data["tb"])
        data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 0)

    # Bit 2: TB threshold (lower range)
    if params["flag_status"][1] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 1)
    else:
        ind = data["tb"] < params["TB_threshold"][0]
        data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 1)

    # Bit 3: TB threshold (upper range)
    if params["flag_status"][2] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 2)
    else:
        ind = data["tb"] > params["TB_threshold"][1]
        data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 2)

    # Bit 4: Spectral consistency threshold
    if params["flag_status"][3] == 1 or "air_pressure" not in data:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 3)
    else:
        try:
            ind = spectral_consistency(data, site, coeff_files)
            data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 3)
        except MissingCoefficientsError as e:
            logging.error(
                f"Error in spectral consistency check: {e}. Proceeding without flagging."
            )
            data["quality_flag_status"] = setbit(data["quality_flag_status"], 3)

    # Bit 5: Receiver sanity
    if params["flag_status"][4] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 4)
    else:
        ind = data["status"] == 1
        data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 4)

    # Bit 6: Rain flag
    if params["flag_status"][5] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 5)
    else:
        ind = data["rain"] == 1
        data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 5)

    # Bit 7: Solar/Lunar flag
    if params["flag_status"][6] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 6)
    else:
        ind = orbpos(data, params)
        data["quality_flag"][ind] = setbit(data["quality_flag"][ind], 6)

    # Bit 8: TB offset threshold
    if params["flag_status"][7] == 1:
        data["quality_flag_status"] = setbit(data["quality_flag_status"], 7)


def orbpos(data: dict, params: dict) -> np.ndarray:
    """Calculates sun & moon elevation/azimuth angles
    and returns index for observations in the direction of the sun.
    """
    time = np.array(
        [
            datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
            for t in data["time"]
        ]
    )
    lat = data["latitude"]
    lng = data["longitude"]
    sol = suncalc.get_position(time, lat=lat, lng=lng)
    lun = suncalc.suncalc.getMoonPosition(time, lat=lat, lng=lng)
    sun = {
        "azimuth_angle": np.rad2deg(sol["azimuth"]) + 180,
        "elevation_angle": np.rad2deg(sol["altitude"]),
    }
    moon = {
        "azimuth_angle": np.rad2deg(lun["azimuth"]) + 180,
        "elevation_angle": np.rad2deg(lun["altitude"]),
    }

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

    flag_ind = (
        (~ma.getmaskarray(data["elevation_angle"]))
        & (data["elevation_angle"] <= np.max(sun["elevation_angle"]) + 10.0)
        & (data["time"] >= sun["rise"])
        & (data["time"] <= sun["set"])
        & (data["elevation_angle"] >= sun["elevation_angle"] - params["saf"])
        & (data["elevation_angle"] <= sun["elevation_angle"] + params["saf"])
        & (
            data["azimuth_angle"]
            >= sun["azimuth_angle"]
            - params["saf"] / np.cos(np.deg2rad(data["elevation_angle"]))
        )
        & (
            data["azimuth_angle"]
            <= sun["azimuth_angle"]
            + params["saf"] / np.cos(np.deg2rad(data["elevation_angle"]))
        )
    ) | (
        (data["elevation_angle"] <= np.max(moon["elevation_angle"]) + 10.0)
        & (data["time"] >= moon["rise"])
        & (data["time"] <= moon["set"])
        & (data["elevation_angle"] >= moon["elevation_angle"] - params["saf"])
        & (data["elevation_angle"] <= moon["elevation_angle"] + params["saf"])
        & (
            data["azimuth_angle"]
            >= moon["azimuth_angle"]
            - params["saf"] / np.cos(np.deg2rad(data["elevation_angle"]))
        )
        & (
            data["azimuth_angle"]
            <= moon["azimuth_angle"]
            + params["saf"] / np.cos(np.deg2rad(data["elevation_angle"]))
        )
    )

    return flag_ind


def spectral_consistency(
    data: dict, site: str | None, coeff_files: Sequence[str | PathLike] | None
) -> np.ndarray:
    """Applies spectral consistency coefficients for given frequency index,
    writes 2S02 product and returns indices to be flagged.
    """
    flag_ind = np.zeros(data["tb"].shape, dtype=np.bool_)
    abs_diff = ma.masked_all(data["tb"].shape, dtype=np.float32)
    data["tb_spectrum"] = ma.masked_all(data["tb"].shape)

    cos_ele = np.ones(len(data["time"]), np.float32)
    cos_ele[(data["elevation_angle"] < 89.5) | (data["pointing_flag"] == 1)] = (
        np.cos(
            np.deg2rad(
                data["elevation_angle"][
                    (data["elevation_angle"] < 89.5) | (data["pointing_flag"] == 1)
                ]
            )
        )
        + 1.0
    )

    prefix = "ins"
    c_list = get_coeff_list(site, prefix, coeff_files)
    if len(c_list) == 0:
        prefix = "spc"
        c_list = get_coeff_list(site, prefix, coeff_files)

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
        ) = get_mvr_coeff(site, prefix, data["frequency"][:], coeff_files)
        ret_in = retrieval_input(data, coeff)
        ele_ind = np.where(
            (np.abs(data["elevation_angle"][:] - 90.0) < 0.5)
            & (data["pointing_flag"][:] == 0)
            & (
                np.abs(
                    data["azimuth_angle"][:] - np.ma.median(data["azimuth_angle"][:])
                )
                < 0.5
            )
        )[0]
        ret_rm = rm_retrieval(data["elevation_angle"][:], coeff, data["frequency"][:])

        _, freq_ind, coeff_ind = np.intersect1d(
            data["frequency"], coeff["AL"], return_indices=True
        )
        c_w1, c_w2, fac = (
            weights1(data["elevation_angle"][:]),
            weights2(data["elevation_angle"][:]),
            factor(data["elevation_angle"][:]),
        )
        in_sc, in_os = (
            input_scale(data["elevation_angle"][:]),
            input_offset(data["elevation_angle"][:]),
        )
        op_sc, op_os = (
            output_scale(data["elevation_angle"][:]),
            output_offset(data["elevation_angle"][:]),
        )

        ret_in[:, 1:] = (ret_in[:, 1:] - in_os) * in_sc
        hidden_layer = np.ones((len(ret_rm), c_w1.shape[2] + 1), np.float32)
        hidden_layer[:, 1:] = np.tanh(
            fac[:].reshape((len(ret_rm), 1))
            * np.einsum("ijk,ij->ik", c_w1, ret_in[:, :])
        )
        data["tb_spectrum"][:, freq_ind] = (
            np.tanh(
                fac[:].reshape((len(ret_rm), 1))
                * np.einsum("ijk,ik->ij", c_w2[:, coeff_ind, :], hidden_layer)
            )
            * op_sc[:, coeff_ind]
            + op_os[:, coeff_ind]
        )

        for ifreq, _ in enumerate(data["frequency"]):
            tb_z = pd.DataFrame(
                {
                    "Tb": np.abs(
                        data["tb"][ele_ind, ifreq] - data["tb_spectrum"][ele_ind, ifreq]
                    )
                },
                index=pd.to_datetime(data["time"][ele_ind], unit="s"),
            )
            tbz_mean = tb_z.resample(
                "10min", origin="start", closed="left", label="left", offset="5min"
            ).mean()
            tbz_mean = tbz_mean.reindex(tb_z.index, method="nearest")
            tb_diff = tb_z - tbz_mean
            tb_diff = tb_diff.reindex(
                pd.to_datetime(data["time"][:], unit="s"), method="nearest"
            )
            tb_diff.iloc[~np.isin(range(len(ret_rm)), ele_ind)] = np.nan
            f_mean = [25.0, 55.0, 90.0, 185.0]  # mean frequency for each band
            fact = [2.0, 4.0, 3.0, 2.5]  # factor for receiver retrieval uncertainty
            # flag for individual channels based on channel retrieval uncertainty
            flag_ind[
                np.where(
                    np.abs(tb_diff["Tb"].values)
                    > ret_rm[:, ifreq]
                    * fact[
                        np.argmin(
                            np.abs(
                                f_mean
                                - np.mean(
                                    data["frequency"][
                                        data["receiver"] == data["receiver"][ifreq]
                                    ]
                                )
                            )
                        )
                    ]
                )[0],
                ifreq,
            ] = True
            abs_diff[ele_ind, ifreq] = np.abs(
                data["tb"][ele_ind, ifreq] - data["tb_spectrum"][ele_ind, ifreq]
            )

            # flag for elevation scans (only receiver 2)
            ele_s = ele_retrieval(
                data["elevation_angle"][:], data["pointing_flag"][:], coeff
            )
            if (len(ele_s) > 0) and (data["receiver"][ifreq] == 2):
                tb_diff = pd.DataFrame(
                    {
                        "Tb": np.abs(
                            data["tb"][ele_s, ifreq] - data["tb_spectrum"][ele_s, ifreq]
                        ),
                    },
                    index=pd.to_datetime(data["time"][ele_s], unit="s"),
                )
                tb_diff = tb_diff[~tb_diff.index.duplicated()].reindex(
                    pd.to_datetime(data["time"][:], unit="s"), method="nearest"
                )
                tb_diff.loc[
                    (data["elevation_angle"][:] > 89.5)
                    | (data["pointing_flag"][:] == 0)
                ] = np.nan
                # flag for individual channels based on channel retrieval uncertainty
                flag_ind[
                    np.where(
                        np.abs(tb_diff["Tb"].values)
                        > ret_rm[:, ifreq] * 2.5 * cos_ele[:]
                    )[0],
                    ifreq,
                ] = True

    else:
        c_list = get_coeff_list(site, "tbx", coeff_files)

        if not c_list:
            raise MissingCoefficientsError("No coefficients found")

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
                        "10min",
                        origin="start",
                        closed="left",
                        label="left",
                        offset="5min",
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
                    ] = True

            abs_diff[:, ifreq] = ma.masked_invalid(
                np.abs(data["tb"][:, ifreq] - data["tb_spectrum"][:, ifreq])
            )

    # receiver flag based on mean absolute difference
    f_mean = [25.0, 55.0, 90.0, 185.0]  # mean frequency for each band
    fact = [1.5, 2.0, 2.0, 3.0]  # factor for receiver retrieval uncertainty
    for _, rec in enumerate(data["receiver_nb"]):
        flag_ind[
            np.ix_(
                ma.mean(abs_diff[:, data["receiver"] == rec], axis=1)
                > fact[
                    np.argmin(
                        np.abs(
                            f_mean - np.mean(data["frequency"][data["receiver"] == rec])
                        )
                    )
                ],
                data["receiver"] == rec,
            )
        ] = True

    return flag_ind


def ele_retrieval(ele_obs: np.ndarray, pointing: np.ndarray, coeff: dict) -> np.ndarray:
    """Extracts elevation angles used in retrieval."""
    ele_ret = coeff["AG"]
    if ele_ret.shape == ():
        ele_ret = np.array([ele_ret])
    return np.array(
        [
            i
            for i, v in enumerate(ele_obs)
            if np.min(np.abs(v - ele_ret)) < 0.5 and pointing[i] == 1
        ]
    )


def rm_retrieval(ele_obs: np.ndarray, coeff: dict, freq) -> np.ndarray:
    """Extracts retrieval uncertainty."""
    rm_ret = coeff["RM"]
    freq_ind = np.array([(np.abs(coeff["AL"] - v)).argmin() for v in freq])
    if rm_ret.shape == ():
        rm_ret = np.array([rm_ret])
    ele_ret = coeff["AG"]
    if ele_ret.shape == ():
        ele_ret = np.array([ele_ret])
    return np.array([rm_ret[freq_ind, np.argmin(np.abs(v - ele_ret))] for v in ele_obs])
