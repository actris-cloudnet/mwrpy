"""Module for writing Level 2 netCDF files"""
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pytz
from timezonefinder import TimezoneFinder

from mwrpy import rpg_mwr
from mwrpy.atmos import eq_pot_tem, pot_tem, rel_hum
from mwrpy.level2.get_ret_coeff import get_mvr_coeff
from mwrpy.level2.lev2_meta_nc import get_data_attributes
from mwrpy.level2.lwp_offset import correct_lwp_offset
from mwrpy.utils import (
    add_time_bounds,
    get_coeff_list,
    get_ret_ang,
    get_ret_freq,
    interpol_2d,
    read_yaml_config,
)

Fill_Value_Float = -999.0
Fill_Value_Int = -99


def lev2_to_nc(
        site: str,
        data_type: str,
        lev1_path: str,
        output_file: str,
        temp_file: str | None = None,
        hum_file: str | None = None,
):
    """This function reads Level 1 files,
    applies retrieval coefficients for Level 2 products
    and writes it into a netCDF file.

    Args:
        site: Name of site.
        data_type: Data type of the netCDF file.
        lev1_path: Path of Level 1 file.
        output_file: Name of output file.
        temp_file: Name of temperature product file.
        hum_file: Name of humidity product file.

    """

    if data_type not in (
            "2P01",
            "2P02",
            "2P03",
            "2P04",
            "2P07",
            "2P08",
            "2I01",
            "2I02",
    ):
        raise RuntimeError(
            ["Data type " + data_type + " not supported for file writing."]
        )

    with nc.Dataset(lev1_path) as lev1:
        if data_type == "2P02":
            bl_scan = _test_bl_scan(site, lev1)
            if not bl_scan:
                data_type = "2P01"
        if data_type in ("2P04", "2P07", "2P08"):
            bl_scan = _test_bl_scan(site, lev1)
            t_product = "2P02"
            if not bl_scan:
                t_product = "2P01"
            for d_type in [t_product, "2P03"]:
                global_attributes, params = read_yaml_config(site)
                rpg_dat, coeff, index = get_products(
                    site, lev1, d_type, params, temp_file=temp_file, hum_file=hum_file
                )
                _combine_lev1(lev1, rpg_dat, index, d_type, params)
                hatpro = rpg_mwr.Rpg(rpg_dat)
                hatpro.data = get_data_attributes(hatpro.data, d_type)
                rpg_mwr.save_rpg(hatpro, output_file, global_attributes, d_type)

        global_attributes, params = read_yaml_config(site)
        rpg_dat, coeff, index = get_products(
            site, lev1, data_type, params, temp_file=temp_file, hum_file=hum_file
        )
        _combine_lev1(lev1, rpg_dat, index, data_type, params)
        _add_att(global_attributes, coeff)
        hatpro = rpg_mwr.Rpg(rpg_dat)
        hatpro.data = get_data_attributes(hatpro.data, data_type)
        rpg_mwr.save_rpg(hatpro, output_file, global_attributes, data_type)


def get_products(
        site: str,
        lev1: nc.Dataset,
        data_type: str,
        params: dict,
        temp_file: str | None = None,
        hum_file: str | None = None,
) -> tuple[dict, dict, np.ndarray]:
    """Derive specified Level 2 products."""

    if "elevation_angle" in lev1.variables:
        elevation_angle = lev1["elevation_angle"][:]
    else:
        elevation_angle = 90 - lev1["zenith_angle"][:]

    rpg_dat, coeff, index = {}, {}, np.empty(0)

    if data_type in ("2I01", "2I02"):
        if data_type == "2I01":
            product = "lwp"
        else:
            product = "iwv"

        coeff = get_mvr_coeff(site, product, lev1["frequency"][:])
        if coeff[0]["RT"] < 2:
            coeff, offset, lin, quad = get_mvr_coeff(
                site, product, lev1["frequency"][:]
            )
        else:
            (
                coeff,
                input_scale,
                input_offset,
                output_scale,
                output_offset,
                weights1,
                weights2,
                factor,
            ) = get_mvr_coeff(site, product, lev1["frequency"][:])
        ret_in = retrieval_input(lev1, coeff)

        index = np.where(
            (lev1["pointing_flag"][:] == 0)
            & np.any(
                np.abs(
                    (np.ones((len(elevation_angle[:]), len(coeff["AG"]))) * coeff["AG"])
                    - np.transpose(
                        np.ones((len(coeff["AG"]), len(elevation_angle[:]))) * elevation_angle[:]
                    )
                )
                < 0.5,
                axis=1,
            )
        )[0]
        if len(index) == 0:
            raise RuntimeError(
                ["No suitable data found for processing for data type: " + data_type]
            )
        coeff["retrieval_elevation_angles"] = str(
            np.sort(np.unique(ele_retrieval(elevation_angle[index], coeff)))
        )
        coeff["retrieval_frequencies"] = str(
            np.sort(np.unique(coeff["FR"]))
        )

        if coeff["RT"] < 2:
            coeff_offset = offset(elevation_angle[index])
            coeff_lin = lin(elevation_angle[index])
            coeff_quad = quad(elevation_angle[index])
            tmp_product = (
                    coeff_offset[:]
                    + np.einsum("ij,ij->i", ret_in[index, :], coeff_lin)
                    + np.einsum("ij,ij->i", ret_in[index, :] ** 2, coeff_quad)
            )

        else:
            c_w1, c_w2, fac = (
                weights1(elevation_angle[index]),
                weights2(elevation_angle[index]),
                factor(elevation_angle[index]),
            )
            in_sc, in_os = input_scale(elevation_angle[index]), input_offset(elevation_angle[index])
            op_sc, op_os = output_scale(elevation_angle[index]), output_offset(elevation_angle[index])

            ret_in[index, 1:] = (ret_in[index, 1:] - in_os[:, :]) * in_sc[:, :]
            hidden_layer = np.ones((len(index), c_w1.shape[2] + 1), np.float32)
            hidden_layer[:, 1:] = np.tanh(fac[:] * np.einsum("ijk,ij->ik", c_w1, ret_in[index, :]))
            tmp_product = np.squeeze(
                np.tanh(fac[:] * np.einsum("ij,ij->i", hidden_layer, c_w2).reshape((len(index), 1))) * op_sc + op_os
            )

        if product == "lwp":
            freq_31 = np.where(np.round(lev1["frequency"][:], 1) == 31.4)[0]
            if len(freq_31) != 1:
                rpg_dat["lwp"], rpg_dat["lwp_offset"] = (
                    tmp_product,
                    np.ones(len(index)) * Fill_Value_Float,
                )
            else:
                rpg_dat["lwp"], rpg_dat["lwp_offset"] = correct_lwp_offset(
                    lev1.variables, tmp_product, index
                )
        else:
            rpg_dat["iwv"] = tmp_product

    elif data_type in ("2P01", "2P03"):
        if data_type == "2P01":
            product, ret = "temperature", "tpt"
        else:
            product, ret = "water_vapor_vmr", "hpt"

        coeff = get_mvr_coeff(site, ret, lev1["frequency"][:])
        if coeff[0]["RT"] < 2:
            coeff, offset, lin, quad = get_mvr_coeff(
                site, ret, lev1["frequency"][:]
            )
        else:
            (
                coeff,
                input_scale,
                input_offset,
                output_scale,
                output_offset,
                weights1,
                weights2,
                factor,
            ) = get_mvr_coeff(site, ret, lev1["frequency"][:])

        ret_in = retrieval_input(lev1, coeff)

        index = np.where(
            (lev1["pointing_flag"][:] == 0)
            & np.any(
                np.abs(
                    (np.ones((len(elevation_angle[:]), len(coeff["AG"]))) * coeff["AG"])
                    - np.transpose(
                        np.ones((len(coeff["AG"]), len(elevation_angle[:]))) * elevation_angle[:]
                    )
                )
                < 0.5,
                axis=1,
            )
        )[0]
        if len(index) == 0:
            raise RuntimeError(
                ["No suitable data found for processing for data type: " + data_type]
            )
        coeff["retrieval_elevation_angles"] = str(
            np.sort(np.unique(ele_retrieval(elevation_angle[index], coeff)))
        )
        coeff["retrieval_frequencies"] = str(
            np.sort(np.unique(coeff["FR"]))
        )

        rpg_dat["altitude"] = coeff["AL"][:] + params["station_altitude"]

        if coeff["RT"] < 2:
            coeff_offset = offset(elevation_angle[index])
            coeff_lin = lin(elevation_angle[index])
            coeff_quad = quad(elevation_angle[index])
            rpg_dat[product] = (
                    coeff_offset
                    + np.einsum("ijk,ik->ij", coeff_lin, ret_in[index, :])
                    + np.einsum("ijk,ik->ij", coeff_quad, ret_in[index, :] ** 2)
            )
            if (coeff["RT"] == 1) & (data_type == "2P03"):
                rpg_dat[product][:, :] = rpg_dat[product][:, :] / 1000.

        else:
            c_w1, c_w2, fac = (
                weights1(elevation_angle[index]),
                weights2(elevation_angle[index]),
                factor(elevation_angle[index]),
            )
            in_sc, in_os = input_scale(elevation_angle[index]), input_offset(elevation_angle[index])
            op_sc, op_os = output_scale(elevation_angle[index]), output_offset(elevation_angle[index])

            ret_in[index, 1:] = (ret_in[index, 1:] - in_os) * in_sc
            hidden_layer = np.ones((len(index), c_w1.shape[2] + 1), np.float32)
            hidden_layer[:, 1:] = np.tanh(
                fac[:].reshape((len(index), 1)) * np.einsum("ijk,ij->ik", c_w1, ret_in[index, :]))
            rpg_dat[product] = (
                    np.tanh(
                        fac[:].reshape((len(index), 1)) * np.einsum("ijk,ik->ij", c_w2, hidden_layer)) * op_sc + op_os
            )
            if product == "water_vapor_vmr":
                rpg_dat[product] = rpg_dat[product] / 1000.0

    elif data_type == "2P02":

        coeff = get_mvr_coeff(site, "tpb", lev1["frequency"][:])
        if coeff[0]["RT"] < 2:
            coeff, offset, lin, quad = get_mvr_coeff(
                site, "tpb", lev1["frequency"][:]
            )
        else:
            coeff, _, _, _, _, _, _, _ = get_mvr_coeff(site, "tpb", lev1["frequency"][:])

        coeff["AG"] = np.sort(coeff["AG"])
        _, freq_ind, _ = np.intersect1d(
            lev1["frequency"][:],
            coeff["FR"],
            assume_unique=False,
            return_indices=True,
        )
        _, freq_bl, _ = np.intersect1d(
            coeff["FR"], coeff["FR_BL"], assume_unique=False, return_indices=True
        )
        coeff["retrieval_frequencies"] = str(
            np.sort(np.unique(coeff["FR"]))
        )

        ix0 = np.where(
            (elevation_angle[:] > coeff["AG"][0] - 0.5)
            & (elevation_angle[:] < coeff["AG"][0] + 0.5)
            & (lev1["pointing_flag"][:] == 1)
            & (np.arange(len(lev1["time"])) + len(coeff["AG"]) < len(lev1["time"]))
        )[0]
        ibl, tb = (
            np.empty([0, len(coeff["AG"])], np.int32),
            np.ones((len(freq_ind), len(coeff["AG"]), 0), np.float32) * Fill_Value_Float,
        )

        for ix0v in ix0:
            if (ix0v + len(coeff["AG"]) < len(lev1["time"])) & (
                    np.allclose(elevation_angle[ix0v: ix0v + len(coeff["AG"])], coeff["AG"], atol=0.5)):
                ibl = np.append(ibl, [np.array(range(ix0v, ix0v + len(coeff["AG"])))], axis=0)
                tb = np.concatenate(
                    (
                        tb,
                        np.expand_dims(lev1["tb"][ix0v: ix0v + len(coeff["AG"]), freq_ind].T, 2),
                    ),
                    axis=2,
                )

        if len(ibl) == 0:
            raise RuntimeError(
                ["No suitable data found for processing for data type: " + data_type]
            )

        index = ibl[:, -1]
        rpg_dat["altitude"] = coeff["AL"][:] + params["station_altitude"]

        if coeff["RT"] < 2:
            tb_alg = []
            if len(freq_ind) - len(freq_bl) > 0:
                tb_alg = np.squeeze(tb[0: len(freq_ind) - len(freq_bl), 0, :])
            for ifq, _ in enumerate(coeff["FR_BL"]):
                if len(tb_alg) == 0:
                    tb_alg = np.squeeze(tb[freq_bl[ifq], :, :])
                else:
                    tb_alg = np.append(tb_alg, np.squeeze(tb[freq_bl[ifq], :, :]), axis=0)

            rpg_dat["temperature"] = offset + np.einsum("jk,ij->ik", lin, tb_alg.T)

        else:
            ret_in = retrieval_input(lev1, coeff)
            ret_array = np.reshape(tb, (len(coeff["AG"]) * len(freq_ind), len(ibl)), "F")
            ret_array = np.concatenate((np.ones((1, len(ibl)), np.float32), ret_array))
            for i_add in range(ret_in.shape[1] - len(coeff["FR"]) - 1, 0, -1):
                ret_array = np.concatenate((ret_array, [ret_in[ibl[:, 0], -i_add]]))
            ret_array[1:, :] = (ret_array[1:, :] - coeff["input_offset"][:, np.newaxis]) * coeff["input_scale"][:,
                                                                                           np.newaxis]
            hidden_layer = np.tanh(coeff["NP"][:] * np.einsum("ij,ik->kj", coeff["W1"], ret_array))
            hidden_layer = np.concatenate((np.ones((len(ibl), 1), np.float32), hidden_layer), axis=1)
            rpg_dat["temperature"] = np.transpose(
                np.tanh(coeff["NP"][:] * np.einsum("ij,kj->ik", coeff["W2"], hidden_layer))
                * coeff["output_scale"][:, np.newaxis] + coeff["output_offset"][:, np.newaxis]
            )

    elif data_type in ("2P04", "2P07", "2P08"):

        tem_dat, tem_freq, tem_ang = load_product(temp_file)
        hum_dat, hum_freq, hum_ang = load_product(hum_file)

        coeff, index = {}, []
        coeff["retrieval_frequencies"] = str(
            np.unique(np.sort(np.concatenate([tem_freq, hum_freq])))
        )
        coeff["retrieval_elevation_angles"] = str(
            np.unique(np.sort(np.concatenate([tem_ang, hum_ang])))
        )
        coeff["retrieval_description"] = "derived product from: " + temp_file + ", " + hum_file
        coeff["dependencies"] = temp_file + ", " + hum_file

        hum_int = interpol_2d(
            hum_dat.variables["time"][:],
            hum_dat.variables["water_vapor_vmr"][:, :],
            tem_dat.variables["time"][:],
        )

        rpg_dat["altitude"] = tem_dat.variables["altitude"][:]
        pres = np.interp(tem_dat.variables["time"][:], lev1["time"][:], lev1["air_pressure"][:])
        if data_type == "2P04":
            rpg_dat["relative_humidity"] = rel_hum(tem_dat.variables["temperature"][:, :], hum_int)
        if data_type == "2P07":
            rpg_dat["potential_temperature"] = pot_tem(
                tem_dat.variables["temperature"][:, :], hum_int, pres, rpg_dat["altitude"]
            )
        if data_type == "2P08":
            rpg_dat["equivalent_potential_temperature"] = eq_pot_tem(
                tem_dat.variables["temperature"][:, :], hum_int, pres, rpg_dat["altitude"]
            )

        _combine_lev1(
            tem_dat.variables,
            rpg_dat,
            np.arange(len(tem_dat.variables["time"][:])),
            data_type,
            params,
        )
    return rpg_dat, coeff, index


def _combine_lev1(
        lev1: nc.Dataset,
        rpg_dat: dict,
        index: np.ndarray,
        data_type: str,
        params: dict,
) -> None:
    """add level1 data"""
    lev1_vars = [
        "time",
        "time_bnds",
        "azimuth_angle",
        "elevation_angle",
        "zenith_angle",
        "altitude",
        "latitude",
        "longitude",
    ]
    if index:
        for ivars in lev1_vars:
            if ivars not in lev1.variables:
                continue
            if (ivars == "time_bnds") & (data_type == "2P02"):
                rpg_dat[ivars] = add_time_bounds(
                    lev1["time"][index], params["scan_time"]
                )
            elif (ivars == "time_bnds") & (data_type in ("2P04", "2P07", "2P08")):
                rpg_dat[ivars] = np.ones(lev1[ivars].shape, np.int32) * Fill_Value_Int
            else:
                try:
                    rpg_dat[ivars] = lev1[ivars][index]
                except IndexError:
                    rpg_dat[ivars] = lev1[ivars][:]


def _add_att(global_attributes: dict, coeff: dict) -> None:
    "add retrieval and calibration attributes"
    fields = [
        "retrieval_type",
        "retrieval_elevation_angles",
        "retrieval_frequencies",
        "retrieval_auxiliary_input",
        "retrieval_description",
    ]
    for name in fields:
        if name in coeff:
            global_attributes[name] = coeff[name]
        else:
            global_attributes[name] = ""

    # remove lev1 only attributes
    att_del = ["ir_instrument", "met_instrument", "_accuracy"]
    att_names = global_attributes.keys()
    for name in list(att_names):
        if any(x in name for x in att_del):
            del global_attributes[name]


def load_product(filename: str):
    """load existing lev2 file for deriving other products"""
    file = nc.Dataset(filename)
    ret_freq = get_ret_freq(filename)
    ret_ang = get_ret_ang(filename)
    return file, ret_freq, ret_ang


def _test_bl_scan(site: str, lev1: nc.Dataset) -> bool:
    """Check for existing BL scans in lev1 data"""

    if "elevation_angle" in lev1.variables:
        elevation_angle = lev1["elevation_angle"][:]
    else:
        elevation_angle = 90 - lev1["zenith_angle"][:]

    bl_scan = True
    coeff_file = get_coeff_list(site, "tpb")
    if len(coeff_file) > 0:
        coeff = get_mvr_coeff(site, "tpb", lev1["frequency"][:])
        bl_ind = np.where(
            (elevation_angle[:] > coeff[0]["AG"][0] - 0.5)
            & (elevation_angle[:] < coeff[0]["AG"][0] + 0.5)
            & (lev1["pointing_flag"][:] == 1)
        )[0]
        if len(bl_ind) == 0:
            bl_scan = False
    else:
        bl_scan = False
    return bl_scan


def ele_retrieval(ele_obs: np.ndarray, coeff: dict) -> np.ndarray:
    """Extracts elevation angles used in retrieval"""
    ele_ret = coeff["AG"]
    if ele_ret.shape == ():
        ele_ret = np.array([ele_ret])
    return np.array([ele_ret[(np.abs(ele_ret - v)).argmin()] for v in ele_obs])


def retrieval_input(lev1: dict | nc.Dataset, coeff: dict) -> np.ndarray:
    """Get retrieval input"""

    time_median = np.nanmedian(lev1["time"][:])[0]
    if time_median < 24:
        date = [lev1.year, lev1.month, lev1.day]
        time_median = decimal_hour2unix(date, time_median)

    _, freq_ind, _ = np.intersect1d(
        lev1["frequency"][:],
        coeff["FR"][:],
        assume_unique=False,
        return_indices=True,
    )
    bias = np.ones((len(lev1["time"][:]), 1), np.float32)

    if coeff["RT"] == Fill_Value_Int:
        ret_in = lev1["tb"][:, :]
    elif coeff["RT"] in (0, 1):
        ret_in = lev1["tb"][:, freq_ind]
    else:
        ret_in = np.concatenate((bias, lev1["tb"][:, freq_ind]), axis=1)

        if coeff.get("TS")[0] == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["air_temperature"][:], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        if coeff.get("HS")[0] == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["relative_humidity"][:], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        if coeff.get("PS")[0] == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["air_pressure"][:], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        if coeff.get("ZS") == 1:
            ret_in = np.concatenate((ret_in, lev1["irt"][:, :]), axis=1)
        if coeff.get("IR") == 1:
            ret_in = np.concatenate((ret_in, lev1["irt"][:, :]), axis=1)
        if coeff.get("I1") == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["irt"][:, 0], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        if coeff.get("I2") == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["irt"][:, 1], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        if coeff.get("DY") == 1:
            doy = np.ones((len(lev1["time"][:]), 2), np.float32) * Fill_Value_Float
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(
                lng=np.nanmedian(lev1["station_longitude"])[0], lat=np.nanmedian(lev1["station_latitude"])[0]
            )
            timezone = pytz.timezone(timezone_str)
            dtime = datetime.fromtimestamp(time_median, timezone)
            dyear = datetime(dtime.year, 12, 31, 0, 0).timetuple().tm_yday
            doy[:, 0] = np.cos(
                datetime.fromtimestamp(time_median).timetuple().tm_yday / dyear * 2 * np.pi)
            doy[:, 1] = np.sin(
                datetime.fromtimestamp(time_median).timetuple().tm_yday / dyear * 2 * np.pi)
            ret_in = np.concatenate((ret_in, doy), axis=1)
        if coeff.get("SU") == 1:
            sun = np.ones((len(lev1["time"][:]), 2), np.float32) * Fill_Value_Float
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(
                lng=np.nanmedian(lev1["station_longitude"])[0], lat=np.nanmedian(lev1["station_latitude"])[0]
            )
            timezone = pytz.timezone(timezone_str)
            dtime = datetime.fromtimestamp(time_median, timezone)
            sun[:, 0] = np.cos(
                (dtime.hour + dtime.minute / 60 + dtime.second / 3600) / 24 * 2 * np.pi
            )
            sun[:, 1] = np.sin(
                (dtime.hour + dtime.minute / 60 + dtime.second / 3600) / 24 * 2 * np.pi
            )
            ret_in = np.concatenate((ret_in, sun), axis=1)

    return ret_in


def decimal_hour2unix(date: list, time: np.ndarray) -> np.ndarray:
    unix_timestamp = np.datetime64("-".join(date)).astype("datetime64[s]").astype("int")
    return (time * 60 * 60 + unix_timestamp).astype(int)
