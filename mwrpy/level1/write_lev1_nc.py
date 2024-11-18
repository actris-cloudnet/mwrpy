"""Module for writing Level 1 netCDF files."""

import logging
from collections.abc import Callable
from itertools import groupby
from typing import TypeAlias

import numpy as np
from numpy import ma

from mwrpy import atmos, rpg_mwr
from mwrpy.exceptions import MissingInputData
from mwrpy.level1.lev1_meta_nc import get_data_attributes
from mwrpy.level1.met_quality_control import apply_met_qc
from mwrpy.level1.quality_control import apply_qc
from mwrpy.level1.rpg_bin import RpgBin
from mwrpy.utils import (
    add_interpol1d,
    add_time_bounds,
    get_file_list,
    isbit,
    read_config,
    update_lev1_attributes,
)

FuncType: TypeAlias = Callable[[str], np.ndarray]


def lev1_to_nc(
    data_type: str,
    path_to_files: str,
    site: str | None = None,
    output_file: str | None = None,
    coeff_files: list | None = None,
    instrument_config: dict | None = None,
) -> rpg_mwr.Rpg:
    """This function reads one day of RPG MWR binary files,
    adds attributes and writes it into netCDF file.

    Args:
        data_type: Data type of the netCDF file.
        path_to_files: Folder containing one day of RPG MWR binary files.
        site: Name of site.
        output_file: Output file name.
        coeff_files: List of coefficient files.
        instrument_config: Dictionary containing information about the instrument.

    Raises:
        MissingInputData: if required input file is missing.
    """
    if site is None:
        assert coeff_files is not None
        assert instrument_config is not None

    if coeff_files is None:
        logging.info(
            f"No coefficient files given, using files in repository for {site}."
        )

    if instrument_config is None:
        logging.info(
            f"No instrument config given, using config file in repository for {site}."
        )

    params = read_config(site, "params")
    if instrument_config is not None:
        params = {**params, **instrument_config}

    rpg_bin = prepare_data(path_to_files, data_type, params)

    if data_type in ("1B01", "1C01"):
        apply_qc(site, rpg_bin, params, coeff_files)
    if data_type in ("1B21", "1C01"):
        apply_met_qc(rpg_bin.data, params)
    hatpro = rpg_mwr.Rpg(rpg_bin.data)
    hatpro.find_valid_times()
    hatpro.data = get_data_attributes(hatpro.data, data_type)
    if output_file is not None:
        global_attributes = read_config(site, "global_specs")
        if data_type != "1C01":
            update_lev1_attributes(global_attributes, data_type)
        rpg_mwr.save_rpg(hatpro, output_file, global_attributes, data_type)
    return hatpro


def prepare_data(
    path_to_files: str,
    data_type: str,
    params: dict,
) -> RpgBin:
    """Load and prepare data for netCDF writing."""
    if data_type in ("1B01", "1C01"):
        brt_files = get_file_list(path_to_files, "BRT")
        if len(brt_files) == 0:
            raise MissingInputData("No BRT files found")
        rpg_bin = RpgBin(brt_files)
        ind_bandwidth = np.argsort(params["bandwidth"])
        rpg_bin.data["tb"] = rpg_bin.data["tb"][:, ind_bandwidth]
        rpg_bin.data["frequency"] = rpg_bin.header["_f"][ind_bandwidth]
        fields = [
            "bandwidth",
            "n_sidebands",
            "sideband_IF_separation",
            "freq_shift",
            "receiver_nb",
            "receiver",
        ]
        for name in fields:
            rpg_bin.data[name] = np.array(params[name])
        rpg_bin.data["time_bnds"] = add_time_bounds(
            rpg_bin.data["time"], params["int_time"]
        )
        rpg_bin.data["pointing_flag"] = np.zeros(len(rpg_bin.data["time"]), np.int32)

        if data_type == "1B01":
            (
                rpg_bin.data["liquid_cloud_flag"],
                rpg_bin.data["liquid_cloud_flag_status"],
            ) = atmos.find_lwcl_free(rpg_bin.data)
        else:
            (
                rpg_bin.data["liquid_cloud_flag"],
                rpg_bin.data["liquid_cloud_flag_status"],
            ) = (
                np.ones(len(rpg_bin.data["time"]), np.int32) * 2,
                np.ones(len(rpg_bin.data["time"]), np.int32),
            )

        file_list_hkd = get_file_list(path_to_files, "HKD")
        if len(file_list_hkd) == 0:
            raise MissingInputData("No HKD files found")
        rpg_hkd = RpgBin(file_list_hkd)
        if (
            "temp" not in rpg_hkd.data
            or "stab" not in rpg_hkd.data
            or "status" not in rpg_hkd.data
        ):
            raise MissingInputData(
                "No temperature, stability or status data in HKD file"
            )

        rpg_bin.data["status"] = np.zeros(
            (len(rpg_bin.data["time"]), len(params["receiver"])), np.int32
        )
        _, ind_brt, ind_hkd = np.intersect1d(
            rpg_bin.data["time"],
            rpg_hkd.data["time"],
            assume_unique=False,
            return_indices=True,
        )
        rpg_bin.data["status"][ind_brt, :] = hkd_sanity_check(
            rpg_hkd.data["status"][ind_hkd], params, rpg_hkd.data["temp"][ind_hkd, 0:2]
        )

        file_list_bls = get_file_list(path_to_files, "BLS")
        if len(file_list_bls) > 0:
            rpg_bls = RpgBin(file_list_bls)
            _add_bls(rpg_bin, rpg_bls, rpg_hkd, params)
        else:
            file_list_blb = get_file_list(path_to_files, "BLB")
            if len(file_list_blb) > 0 and np.any(rpg_hkd.data["status"][:] & 2**18 > 0):
                rpg_blb = RpgBin(file_list_blb)
                _add_blb(rpg_bin, rpg_blb, rpg_hkd, params)

        if params["azi_cor"] != -999.0:
            _azi_correction(rpg_bin.data, params)

        if params["const_azi"] != -999.0:
            rpg_bin.data["azimuth_angle"] = (
                rpg_bin.data["azimuth_angle"] + params["const_azi"]
            ) % 360

        if data_type == "1C01":
            if params["ir_flag"]:
                file_list_irt = get_file_list(path_to_files, "IRT")
                if len(file_list_irt) > 0:
                    rpg_irt = RpgBin(file_list_irt)
                    rpg_irt.data["irt"][rpg_irt.data["irt"] <= 125.5] = ma.masked
                    rpg_bin.data["ir_wavelength"] = rpg_irt.header["_f"] * 1e-6
                    rpg_bin.data["ir_bandwidth"] = params["ir_bandwidth"] * 1e-6
                    rpg_bin.data["ir_beamwidth"] = params["ir_beamwidth"]
                    add_interpol1d(
                        rpg_bin.data, rpg_irt.data["irt"], rpg_irt.data["time"], "irt"
                    )
                    add_interpol1d(
                        rpg_bin.data,
                        rpg_irt.data["ir_elevation_angle"],
                        rpg_irt.data["time"],
                        "ir_elevation_angle",
                    )
                    add_interpol1d(
                        rpg_bin.data,
                        rpg_irt.data["ir_azimuth_angle"],
                        rpg_irt.data["time"],
                        "ir_azimuth_angle",
                    )

            (
                rpg_bin.data["liquid_cloud_flag"],
                rpg_bin.data["liquid_cloud_flag_status"],
            ) = atmos.find_lwcl_free(rpg_bin.data)

            file_list_met = get_file_list(path_to_files, "MET")
            if len(file_list_met) == 0:
                raise MissingInputData("No MET files found")
            rpg_met = RpgBin(file_list_met)
            add_interpol1d(
                rpg_bin.data,
                rpg_met.data["air_temperature"],
                rpg_met.data["time"],
                "air_temperature",
            )
            add_interpol1d(
                rpg_bin.data,
                rpg_met.data["relative_humidity"],
                rpg_met.data["time"],
                "relative_humidity",
            )
            add_interpol1d(
                rpg_bin.data,
                rpg_met.data["air_pressure"] * 100,
                rpg_met.data["time"],
                "air_pressure",
            )
            if "wind_speed" in rpg_met.data:
                add_interpol1d(
                    rpg_bin.data,
                    rpg_met.data["wind_speed"] / 3.6,
                    rpg_met.data["time"],
                    "wind_speed",
                )
            if "wind_direction" in rpg_met.data:
                add_interpol1d(
                    rpg_bin.data,
                    rpg_met.data["wind_direction"],
                    rpg_met.data["time"],
                    "wind_direction",
                )
            if "rainfall_rate" in rpg_met.data:
                add_interpol1d(
                    rpg_bin.data,
                    rpg_met.data["rainfall_rate"] / 1000 / 3600,
                    rpg_met.data["time"],
                    "rainfall_rate",
                )

    elif data_type == "1B11":
        file_list_irt = get_file_list(path_to_files, "IRT")
        rpg_bin = RpgBin(file_list_irt)
        rpg_bin.data["irt"][rpg_bin.data["irt"] <= 125.5] = ma.masked
        rpg_bin.data["ir_wavelength"] = rpg_bin.header["_f"] * 1e-6
        rpg_bin.data["ir_bandwidth"] = params["ir_bandwidth"] * 1e-6
        rpg_bin.data["ir_beamwidth"] = params["ir_beamwidth"]

    elif data_type == "1B21":
        file_list_met = get_file_list(path_to_files, "MET")
        rpg_bin = RpgBin(file_list_met)
        if "wind_speed" in rpg_bin.data:
            rpg_bin.data["wind_speed"] /= 3.6
        if "air_pressure" in rpg_bin.data:
            rpg_bin.data["air_pressure"] *= 100
        if "rainfall_rate" in rpg_bin.data:
            rpg_bin.data["rainfall_rate"] /= 3.6e6

    else:
        raise RuntimeError(
            ["Data type " + data_type + " not supported for file writing."]
        )

    file_list_hkd = get_file_list(path_to_files, "HKD")
    _append_hkd(file_list_hkd, rpg_bin, data_type, params)
    rpg_bin.data["altitude"] = (
        np.ones(len(rpg_bin.data["time"]), np.float32) * params["altitude"]
    )

    return rpg_bin


def _append_hkd(
    file_list_hkd: list, rpg_bin: RpgBin, data_type: str, params: dict
) -> None:
    """Append hkd data on same time grid and perform TB sanity check."""
    hkd = RpgBin(file_list_hkd)

    if "latitude" not in hkd.data:
        add_interpol1d(
            rpg_bin.data,
            np.ones(len(hkd.data["time"])) * params["latitude"],
            hkd.data["time"],
            "latitude",
        )
    else:
        idx = np.where(~ma.getmaskarray(hkd.data["latitude"]))[0]
        add_interpol1d(
            rpg_bin.data,
            hkd.data["latitude"][idx],
            hkd.data["time"][idx],
            "latitude",
        )
    if "longitude" not in hkd.data:
        add_interpol1d(
            rpg_bin.data,
            np.ones(len(hkd.data["time"])) * params["longitude"],
            hkd.data["time"],
            "longitude",
        )
    else:
        idx = np.where(~ma.getmaskarray(hkd.data["longitude"]))[0]
        add_interpol1d(
            rpg_bin.data,
            hkd.data["longitude"][idx],
            hkd.data["time"][idx],
            "longitude",
        )

    if data_type in ("1B01", "1C01"):
        hkd.data["temp"][hkd.data["temp"] >= 350.0] = ma.masked
        add_interpol1d(
            rpg_bin.data, hkd.data["temp"][:, 0:2], hkd.data["time"], "t_amb"
        )
        add_interpol1d(
            rpg_bin.data, hkd.data["temp"][:, 2:4], hkd.data["time"], "t_rec"
        )
        add_interpol1d(rpg_bin.data, hkd.data["stab"], hkd.data["time"], "t_sta")


def hkd_sanity_check(status: np.ndarray, params: dict, t_amb: np.ndarray) -> np.ndarray:
    """Perform sanity checks for .HKD data."""
    t_amb[t_amb >= 350.0] = ma.masked
    status_flag = np.zeros((len(status), len(params["receiver"])), np.int32)
    for irec, nrec in enumerate(params["receiver"]):
        # status flags for individual channels
        status_flag[~isbit(status, irec + (nrec - 1)), irec] = 1
        if nrec == 1:
            # receiver 1 thermal stability & ambient target stability & noise diode
            if np.all(ma.is_masked(t_amb[:, 0])) | np.all(ma.is_masked(t_amb[:, 1])):
                status_flag[
                    isbit(status, 25)
                    | ~isbit(status, 22)
                    | (~isbit(status, 24) & ~isbit(status, 25)),
                    irec,
                ] = 1
            else:
                status_flag[
                    isbit(status, 25)
                    | isbit(status, 29)
                    | ~isbit(status, 22)
                    | (~isbit(status, 24) & ~isbit(status, 25)),
                    irec,
                ] = 1
        if nrec == 2:
            # receiver 2 thermal stability & ambient target stability & noise diode
            if np.all(ma.is_masked(t_amb[:, 0])) | np.all(ma.is_masked(t_amb[:, 1])):
                status_flag[
                    isbit(status, 27)
                    | ~isbit(status, 23)
                    | (~isbit(status, 26) & ~isbit(status, 27)),
                    irec,
                ] = 1
            else:
                status_flag[
                    isbit(status, 27)
                    | isbit(status, 29)
                    | ~isbit(status, 23)
                    | (~isbit(status, 26) & ~isbit(status, 27)),
                    irec,
                ] = 1

    return status_flag


def _find_closest_ind(a, b, threshold):
    """Find indices of the closest values in two arrays within a threshold."""
    closest_ind = np.argmin(np.abs(a - b[:, np.newaxis]), axis=0)
    below_threshold = np.abs(b[closest_ind] - a) <= threshold
    return np.where(below_threshold)[0], closest_ind[below_threshold]


def _add_bls(brt: RpgBin, bls: RpgBin, hkd: RpgBin, params: dict) -> None:
    """Add BLS boundary-layer scans using a linear time axis."""
    bls.data["time_bnds"] = add_time_bounds(bls.data["time"] + 1, params["int_time"])
    bls.data["status"] = np.zeros(
        (len(bls.data["time"]), len(params["receiver"])), np.int32
    )
    time_ind, ind_hkd = _find_closest_ind(
        bls.data["time"], hkd.data["time"], params["int_time"] * 2
    )
    bls.data["status"][time_ind, :] = hkd_sanity_check(
        hkd.data["status"][ind_hkd], params, hkd.data["temp"][ind_hkd, 0:2]
    )
    bls.data["pointing_flag"] = np.ones(len(bls.data["time"]), np.int32)
    bls.data["liquid_cloud_flag"] = np.ones(len(bls.data["time"]), np.int32) * 2
    bls.data["liquid_cloud_flag_status"] = ma.masked_all(
        len(bls.data["time"]), np.int32
    )
    brt.data["time"] = np.concatenate((brt.data["time"], bls.data["time"]))
    ind = np.argsort(brt.data["time"])
    brt.data["time"] = brt.data["time"][ind]

    names = [
        "time_bnds",
        "elevation_angle",
        "azimuth_angle",
        "rain",
        "tb",
        "pointing_flag",
        "liquid_cloud_flag",
        "liquid_cloud_flag_status",
        "status",
    ]
    for var in names:
        brt.data[var] = np.concatenate((brt.data[var], bls.data[var]))
        if brt.data[var].ndim > 1:
            brt.data[var] = brt.data[var][ind, :]
        else:
            brt.data[var] = brt.data[var][ind]
    brt.header["n"] = len(brt.data["time"])


def _add_blb(brt: RpgBin, blb: RpgBin, hkd: RpgBin, params: dict) -> None:
    """Add BLB boundary-layer scans using a linear time axis."""
    time_bnds_add: np.ndarray = np.empty([0], dtype=np.int32)
    time_add: np.ndarray = np.empty([0], dtype=np.int32)
    elevation_angle_add: np.ndarray = np.empty([0], dtype=np.int32)
    azimuth_angle_add: np.ndarray = np.empty([0], dtype=np.int32)
    rain_add: np.ndarray = np.empty([0], dtype=np.int32)
    tb_add: np.ndarray = np.empty([0], dtype=np.int32)
    status_add: np.ndarray = np.empty([0], dtype=np.int32)
    seqs_all = [
        (key, len(list(val))) for key, val in groupby(hkd.data["status"][:] & 2**18 > 0)
    ]
    seqs = np.array(
        [
            (key, sum(s[1] for s in seqs_all[:i]), length)
            for i, (key, length) in enumerate(seqs_all)
            if bool(key) is True
        ]
    )
    scan_time = np.median(seqs[:, 2])

    for time_ind, time_blb in enumerate(blb.data["time"]):
        seqi = np.where(
            np.abs(hkd.data["time"][seqs[:, 1] + seqs[:, 2] - 1] - time_blb) < 60
        )[0]
        if len(seqi) != 1:
            time_blb = time_blb + int(scan_time)
            seqi = np.where(
                np.abs(hkd.data["time"][seqs[:, 1] + seqs[:, 2] - 1] - time_blb) < 60
            )[0]
            if len(seqi) != 1:
                continue

        if (
            np.abs(time_blb - hkd.data["time"][seqs[seqi, 1]][0])
            >= blb.header["_n_ang"]
        ):
            scan_quadrant = 0.0  # scan quadrant, 0 deg: 1st, 180 deg: 2nd
            if (isbit(blb.data["rain"][time_ind], 1)) & (
                not isbit(blb.data["rain"][time_ind], 2)
            ):
                scan_quadrant = 180.0

            time0 = hkd.data["time"][seqs[seqi, 1]][0]
            time_norm = int(np.floor(((time_blb - 1) - time0) / (blb.header["_n_ang"])))
            time_add = np.concatenate(
                (
                    time_add,
                    np.squeeze(
                        np.linspace(
                            time0 + time_norm,
                            time0 + time_norm * blb.header["_n_ang"],
                            blb.header["_n_ang"],
                            dtype=np.int32,
                        )
                    ),
                )
            )

            brt_ind = np.where(
                (brt.data["time"] > time_blb - 3600)
                & (brt.data["time"] < time_blb + 3600)
            )[0]
            brt_azi = ma.median(brt.data["azimuth_angle"][brt_ind])
            azimuth_angle_add = np.concatenate(
                (
                    azimuth_angle_add,
                    np.ones(blb.header["_n_ang"]) * ((scan_quadrant + brt_azi) % 360),
                )
            )

            rain_add = np.concatenate(
                (
                    rain_add,
                    np.ones(blb.header["_n_ang"], np.int32)
                    * int(isbit(blb.data["rain"][time_ind], 0)),
                )
            )
            elevation_angle_add = np.concatenate(
                (elevation_angle_add, blb.header["_ang"])
            )

            for ang in range(blb.header["_n_ang"]):
                if len(tb_add) == 0:
                    tb_add = blb.data["tb"][time_ind, :, ang]
                else:
                    tb_add = np.vstack((tb_add, blb.data["tb"][time_ind, :, ang]))

            if len(time_bnds_add) == 0:
                time_bnds_add = add_time_bounds(
                    time_add,
                    int(np.floor(scan_time / (blb.header["_n_ang"]))),
                )
            else:
                time_bnds_add = np.concatenate(
                    (
                        time_bnds_add,
                        add_time_bounds(
                            time_add,
                            int(np.floor(scan_time / (blb.header["_n_ang"]))),
                        ),
                    )
                )

            blb_status = hkd_sanity_check(
                hkd.data["status"][
                    seqs[seqi, 1][0] : seqs[seqi, 1][0] + seqs[seqi, 2][0]
                ],
                params,
                hkd.data["temp"][
                    seqs[seqi, 1][0] : seqs[seqi, 1][0] + seqs[seqi, 2][0], 0:2
                ],
            )
            blb_status_add = np.zeros(
                (blb.header["_n_ang"], len(params["receiver"])), np.int32
            )
            for i_ch, _ in enumerate(params["receiver"]):
                blb_status_add[:, i_ch] = int(np.any(blb_status[:, i_ch]))
            if len(status_add) == 0:
                status_add = blb_status_add
            else:
                status_add = np.concatenate((status_add, blb_status_add))

    if len(time_add) > 0:
        pointing_flag_add = np.ones(len(time_add), np.int32)
        liquid_cloud_flag_add = np.ones(len(time_add), np.int32) * 2
        liquid_cloud_flag_status_add = ma.masked_all(len(time_add), np.int32)
        brt.data["time"] = np.concatenate((brt.data["time"], time_add))
        ind = np.argsort(brt.data["time"])
        brt.data["time"] = brt.data["time"][ind]

        names_add: dict[str, np.ndarray] = {
            "time_bnds": time_bnds_add,
            "elevation_angle": elevation_angle_add,
            "azimuth_angle": azimuth_angle_add,
            "rain": rain_add,
            "tb": tb_add,
            "pointing_flag": pointing_flag_add,
            "liquid_cloud_flag": liquid_cloud_flag_add,
            "liquid_cloud_flag_status": liquid_cloud_flag_status_add,
            "status": status_add,
        }

        for var in names_add.items():
            brt.data[var[0]] = np.concatenate((brt.data[var[0]], var[1]))
            if brt.data[var[0]].ndim > 1:
                brt.data[var[0]] = brt.data[var[0]][ind, :]
            else:
                brt.data[var[0]] = brt.data[var[0]][ind]
        brt.header["n"] = len(brt.data["time"])


def _azi_correction(brt: dict, params: dict) -> None:
    """Azimuth correction (transform to "geographical" coordinates)."""
    ind180 = np.where((brt["azimuth_angle"][:] >= 0) & (brt["azimuth_angle"][:] <= 180))
    ind360 = np.where(
        (brt["azimuth_angle"][:] > 180) & (brt["azimuth_angle"][:] <= 360)
    )
    brt["azimuth_angle"][ind180] = params["azi_cor"] - brt["azimuth_angle"][ind180]
    brt["azimuth_angle"][ind360] = (
        360.0 + params["azi_cor"] - brt["azimuth_angle"][ind360]
    )
    brt["azimuth_angle"][brt["azimuth_angle"][:] < 0] += 360.0
