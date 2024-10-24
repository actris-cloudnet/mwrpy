"""Module for writing Level 2 netCDF files."""

from datetime import datetime
from itertools import groupby

import netCDF4 as nc
import numpy as np
import pytz
from numpy import ma
from timezonefinder import TimezoneFinder

from mwrpy import rpg_mwr
from mwrpy.atmos import eq_pot_tem, pot_tem, rel_hum
from mwrpy.exceptions import MissingInputData
from mwrpy.level2.get_ret_coeff import get_mvr_coeff
from mwrpy.level2.lev2_meta_nc import get_data_attributes
from mwrpy.level2.lwp_offset import correct_lwp_offset
from mwrpy.utils import (
    interpol_2d,
    interpolate_2d,
    read_config,
)


def lev2_to_nc(
    data_type: str,
    lev1_file: str,
    output_file: str,
    site: str | None = None,
    temp_file: str | None = None,
    hum_file: str | None = None,
    coeff_files: list | None = None,
):
    """This function reads Level 1 files,
    applies retrieval coefficients for Level 2 products
    and writes it into a netCDF file.

    Args:
        data_type: Data type of the netCDF file.
        lev1_file: Path of Level 1 file.
        output_file: Name of output file.
        site: Name of site.
        temp_file: Name of temperature product file.
        hum_file: Name of humidity product file.
        coeff_files: List of coefficient files.

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
        "2I06",
    ):
        raise ValueError(f"Data type {data_type} not recognised")

    global_attributes = read_config(site, "global_specs")
    params = read_config(site, "params")

    with nc.Dataset(lev1_file) as lev1:
        params["altitude"] = np.median(lev1.variables["altitude"][:])

        rpg_dat, coeff, index, scan_time = get_products(
            site,
            lev1,
            data_type,
            params,
            coeff_files=coeff_files,
            temp_file=temp_file,
            hum_file=hum_file,
        )
        _combine_lev1(lev1, rpg_dat, index, data_type, scan_time)
        _del_att(global_attributes)
        hatpro = rpg_mwr.Rpg(rpg_dat)
        hatpro.data = get_data_attributes(hatpro.data, data_type, coeff)
        rpg_mwr.save_rpg(hatpro, output_file, global_attributes, data_type)


def get_products(
    site: str | None,
    lev1: nc.Dataset,
    data_type: str,
    params: dict,
    coeff_files: list | None,
    temp_file: str | None = None,
    hum_file: str | None = None,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Derive specified Level 2 products."""
    if "elevation_angle" in lev1.variables:
        elevation_angle = lev1["elevation_angle"][:]
    else:
        elevation_angle = 90 - lev1["zenith_angle"][:]

    rpg_dat, coeff, index, scan_time = {}, {}, np.empty(0), np.empty(0)

    if data_type in ("2I01", "2I02", "2I06"):
        product = (
            "lwp" if data_type == "2I01" else "iwv" if data_type == "2I02" else "sta"
        )

        coeff = get_mvr_coeff(site, product, lev1["frequency"][:], coeff_files)
        if coeff[0]["RT"] < 2:
            coeff, offset, lin, quad = get_mvr_coeff(
                site, product, lev1["frequency"][:], coeff_files
            )
        else:
            # pylint: disable-next=unbalanced-tuple-unpacking
            (
                coeff,
                input_scale,
                input_offset,
                output_scale,
                output_offset,
                weights1,
                weights2,
                factor,
            ) = get_mvr_coeff(site, product, lev1["frequency"][:], coeff_files)
        ret_in = retrieval_input(lev1, coeff)

        index = np.where(lev1["pointing_flag"][:] == 0)[0]  # type: ignore
        if len(index) == 0:
            raise MissingInputData(
                f"No suitable data found for processing for data type: {data_type}"
            )

        coeff["retrieval_elevation_angles"] = _format_attribute_array(
            ele_retrieval(elevation_angle[index], coeff)
        )
        coeff["retrieval_frequencies"] = _get_retrieval_frequencies(coeff)

        if coeff["RT"] < 2:
            coeff_offset = offset(elevation_angle[index])
            coeff_lin = lin(elevation_angle[index])
            coeff_quad = quad(elevation_angle[index])
            tmp_product = (
                np.squeeze(coeff_offset[:])
                + np.einsum("ij,ij->i", ret_in[index, :], coeff_lin)
                + np.einsum("ij,ij->i", ret_in[index, :] ** 2, coeff_quad)
            )

        else:
            c_w1, c_w2, fac = (
                weights1(elevation_angle[index]),
                weights2(elevation_angle[index]),
                factor(elevation_angle[index]),
            )
            if fac.ndim == 1:
                fac = fac[:, np.newaxis]
            in_sc, in_os = (
                input_scale(elevation_angle[index]),
                input_offset(elevation_angle[index]),
            )
            op_sc, op_os = (
                output_scale(elevation_angle[index]),
                output_offset(elevation_angle[index]),
            )

            ret_in[index, 1:] = (ret_in[index, 1:] - in_os[:, :]) * in_sc[:, :]
            hidden_layer = np.ones((len(index), c_w1.shape[2] + 1), np.float32)
            hidden_layer[:, 1:] = np.tanh(
                fac[:] * np.einsum("ijk,ij->ik", c_w1, ret_in[index, :])
            )
            if product == "sta":
                tmp_product = np.squeeze(
                    np.tanh(fac[:] * np.einsum("ij,ikj->ik", hidden_layer, c_w2))
                    * op_sc
                    + op_os
                )
            else:
                tmp_product = np.squeeze(
                    np.tanh(
                        fac[:]
                        * np.einsum("ij,ij->i", hidden_layer, c_w2).reshape(
                            (len(index), 1)
                        )
                    )
                    * op_sc
                    + op_os
                )

        index_ret = np.where(
            np.any(
                np.abs(
                    (
                        np.ones((len(elevation_angle[index]), len(coeff["AG"])))
                        * coeff["AG"]
                    )
                    - np.transpose(
                        np.ones((len(coeff["AG"]), len(elevation_angle[index])))
                        * elevation_angle[index]
                    )
                )
                <= 0.5,
                axis=1,
            )
        )[0]  # type: ignore
        if product in ("lwp", "iwv"):
            ret_product = ma.masked_all(len(index), np.float32)
            ret_product[index_ret] = tmp_product[index_ret]
            if product == "lwp":
                freq_win = np.where((np.round(lev1["frequency"][:].data, 1) == 31.4))[0]
                rpg_dat["lwp"], rpg_dat["lwp_offset"] = (
                    ret_product,
                    ma.masked_all(len(index)),
                )
                if len(freq_win) == 1 and len(index_ret) > 0:
                    (
                        rpg_dat["lwp"][index_ret],
                        rpg_dat["lwp_offset"][index_ret],
                    ) = correct_lwp_offset(
                        lev1.variables, ret_product[index_ret], index[index_ret]
                    )
            else:
                rpg_dat[product] = ret_product

            _get_qf(rpg_dat, lev1, coeff, index, index_ret, product)

        else:
            product_list = (
                "k_index",
                "ko_index",
                "total_totals",
                "lifted_index",
                "showalter_index",
                "cape",
            )
            for ind, prd in enumerate(product_list):
                ret_product = ma.masked_all(len(index), np.float32)
                ret_product[index_ret] = tmp_product[index_ret, ind]
                rpg_dat[prd] = ret_product

            _get_qf(rpg_dat, lev1, coeff, index, index_ret, "stability")

    elif data_type in ("2P01", "2P03"):
        if data_type == "2P01":
            product, ret = "temperature", "tpt"
        else:
            product, ret = "absolute_humidity", "hpt"

        coeff = get_mvr_coeff(site, ret, lev1["frequency"][:], coeff_files)
        if coeff[0]["RT"] < 2:
            coeff, offset, lin, quad = get_mvr_coeff(
                site, ret, lev1["frequency"][:], coeff_files
            )
        else:
            # pylint: disable-next=unbalanced-tuple-unpacking
            (
                coeff,
                input_scale,
                input_offset,
                output_scale,
                output_offset,
                weights1,
                weights2,
                factor,
            ) = get_mvr_coeff(site, ret, lev1["frequency"][:], coeff_files)

        ret_in = retrieval_input(lev1, coeff)

        index = np.where(lev1["pointing_flag"][:] == 0)[0]  # type: ignore
        if len(index) == 0:
            raise MissingInputData(
                f"No suitable data found for processing for data type: {data_type}"
            )
        coeff["retrieval_elevation_angles"] = _format_attribute_array(
            ele_retrieval(elevation_angle[index], coeff)
        )

        coeff["retrieval_frequencies"] = _get_retrieval_frequencies(coeff)

        rpg_dat["height"] = coeff["AL"][:] + params["altitude"]

        if coeff["RT"] < 2:
            coeff_offset = offset(elevation_angle[index])
            coeff_lin = lin(elevation_angle[index])
            coeff_quad = quad(elevation_angle[index])
            tmp_dat = (
                coeff_offset
                + np.einsum("ijk,ik->ij", coeff_lin, ret_in[index, :])
                + np.einsum("ijk,ik->ij", coeff_quad, ret_in[index, :] ** 2)
            )
            if (coeff["RT"] == 1) and (data_type == "2P03"):
                tmp_dat[:, :] = tmp_dat[:, :] / 1000.0

        else:
            c_w1, c_w2, fac = (
                weights1(elevation_angle[index]),
                weights2(elevation_angle[index]),
                factor(elevation_angle[index]),
            )
            in_sc, in_os = (
                input_scale(elevation_angle[index]),
                input_offset(elevation_angle[index]),
            )
            op_sc, op_os = (
                output_scale(elevation_angle[index]),
                output_offset(elevation_angle[index]),
            )

            ret_in[index, 1:] = (ret_in[index, 1:] - in_os) * in_sc
            hidden_layer = np.ones((len(index), c_w1.shape[2] + 1), np.float32)
            hidden_layer[:, 1:] = np.tanh(
                fac[:].reshape((len(index), 1))
                * np.einsum("ijk,ij->ik", c_w1, ret_in[index, :])
            )
            tmp_dat = (
                np.tanh(
                    fac[:].reshape((len(index), 1))
                    * np.einsum("ijk,ik->ij", c_w2, hidden_layer)
                )
                * op_sc
                + op_os
            )
            if product == "absolute_humidity":
                tmp_dat = tmp_dat / 1000.0

        index_ret = np.where(
            np.any(
                np.abs(
                    (
                        np.ones((len(elevation_angle[index]), len(coeff["AG"])))
                        * coeff["AG"]
                    )
                    - np.transpose(
                        np.ones((len(coeff["AG"]), len(elevation_angle[index])))
                        * elevation_angle[index]
                    )
                )
                <= 0.5,
                axis=1,
            )
        )[0]  # type: ignore
        rpg_dat[product] = ma.masked_all(
            (len(index), len(rpg_dat["height"])), np.float32
        )
        rpg_dat[product][index_ret, :] = tmp_dat[index_ret, :]

        _get_qf(rpg_dat, lev1, coeff, index, index_ret, product)

    elif data_type == "2P02":
        coeff = get_mvr_coeff(site, "tpb", lev1["frequency"][:], coeff_files)
        if coeff[0]["RT"] < 2:
            coeff, offset, lin, quad = get_mvr_coeff(
                site, "tpb", lev1["frequency"][:], coeff_files
            )
        else:
            # pylint: disable-next=unbalanced-tuple-unpacking
            coeff, _, _, _, _, _, _, _ = get_mvr_coeff(
                site, "tpb", lev1["frequency"][:], coeff_files
            )

        coeff["AG"] = np.flip(np.sort(coeff["AG"]))
        _, freq_ind, _ = np.intersect1d(
            lev1["frequency"][:],
            coeff["FR"],
            assume_unique=False,
            return_indices=True,
        )
        _, freq_bl, _ = np.intersect1d(
            coeff["FR"], coeff["FR_BL"], assume_unique=False, return_indices=True
        )
        coeff["retrieval_frequencies"] = _get_retrieval_frequencies(coeff)

        ix0 = np.where(
            (elevation_angle[:] > coeff["AG"][0] - 0.5)
            & (elevation_angle[:] < coeff["AG"][0] + 0.5)
            & (lev1["pointing_flag"][:] == 1)
            & (np.arange(len(lev1["time"])) + len(coeff["AG"]) < len(lev1["time"]))
        )[0]
        ibl, tb, scan_time = (
            np.empty([0, len(coeff["AG"])], np.int32),
            ma.masked_all((len(freq_ind), len(coeff["AG"]), 0), np.float32),
            np.empty(0, np.int32),
        )

        for ix0v in ix0:
            ix1v = ix0v + len(coeff["AG"]) + 10
            ind_multi = np.where(lev1["pointing_flag"][ix0v:ix1v] == 1)[0]
            _, ind_ang, _ = np.intersect1d(
                elevation_angle[ix0v + ind_multi], coeff["AG"], return_indices=True
            )
            if ix1v < len(lev1["time"]) and len(ind_ang) == len(coeff["AG"]):
                scan_time = np.append(
                    scan_time,
                    [
                        np.array(
                            lev1["time"][ix0v + ind_ang[0]]
                            - lev1["time"][ix0v + ind_ang[-1]]
                        )
                    ],
                    axis=0,
                )
                ibl = np.append(ibl, [ix0v + np.flip(ind_ang)], axis=0)
                tb = np.concatenate(
                    (
                        tb,
                        np.expand_dims(
                            lev1["tb"][ix0v + np.flip(ind_ang), freq_ind].T, 2
                        ),
                    ),
                    axis=2,
                )

        if len(ibl) <= 1:
            raise MissingInputData(
                f"No suitable data found for processing for data type: {data_type}"
            )

        index = ibl[:, -1]  # type: ignore
        rpg_dat["height"] = coeff["AL"][:] + params["altitude"]

        if coeff["RT"] < 2:
            tb_alg: np.ndarray = np.array([])
            if len(freq_ind) - len(freq_bl) > 0:
                tb_alg = np.squeeze(tb[0 : len(freq_ind) - len(freq_bl), 0, :])
            for ifq, _ in enumerate(coeff["FR_BL"]):
                if len(tb_alg) == 0:
                    tb_alg = np.squeeze(tb[freq_bl[ifq], :, :])
                else:
                    tb_alg = np.append(
                        tb_alg, np.squeeze(tb[freq_bl[ifq], :, :]), axis=0
                    )

            rpg_dat["temperature"] = np.transpose(offset(0)) + np.einsum(
                "jk,ij->ik", lin(0), np.transpose(tb_alg)
            )

        else:
            ret_in = retrieval_input(lev1, coeff)
            ret_array = np.reshape(
                tb, (len(coeff["AG"]) * len(freq_ind), len(ibl)), order="F"
            )
            ret_array = np.concatenate((np.ones((1, len(ibl)), np.float32), ret_array))
            for i_add in range(ret_in.shape[1] - len(coeff["FR"]) - 1, 0, -1):
                ibl_ind = ibl[:, 0]
                ret_ind = ret_in[ibl_ind, -i_add]
                xx = np.array([ret_ind])
                ret_array = np.concatenate((ret_array, xx))

            ret_array[1:, :] = (
                ret_array[1:, :] - coeff["input_offset"][:, np.newaxis]
            ) * coeff["input_scale"][:, np.newaxis]
            hidden_layer = np.tanh(
                coeff["NP"][:] * np.einsum("ij,ik->kj", coeff["W1"], ret_array)
            )
            hidden_layer = np.concatenate(
                (np.ones((len(ibl), 1), np.float32), hidden_layer), axis=1
            )
            rpg_dat["temperature"] = np.transpose(
                np.tanh(
                    coeff["NP"][:] * np.einsum("ij,kj->ik", coeff["W2"], hidden_layer)
                )
                * coeff["output_scale"][:, np.newaxis]
                + coeff["output_offset"][:, np.newaxis]
            )

        _get_qf(rpg_dat, lev1, coeff, index, np.array(range(len(index))), "temperature")

    elif data_type in ("2P04", "2P07", "2P08"):
        assert temp_file is not None
        assert hum_file is not None
        tem_dat = load_product(temp_file)
        hum_dat = load_product(hum_file)

        coeff, index = {}, np.empty(0, np.int32)

        coeff["retrieval_frequencies"] = _combine_array_attributes(
            tem_dat, hum_dat, "retrieval_frequencies"
        )

        coeff["retrieval_elevation_angles"] = _combine_array_attributes(
            tem_dat, hum_dat, "retrieval_elevation_angles"
        )

        coeff["retrieval_type"] = "derived product"
        coeff["dependencies"] = temp_file + ", " + hum_file
        if len(hum_dat.variables["height"][:]) == len(tem_dat.variables["height"][:]):
            hum_int = interpol_2d(
                hum_dat.variables["time"][:],
                hum_dat.variables["absolute_humidity"][:, :],
                tem_dat.variables["time"][:],
            )
        else:
            hum_int = interpolate_2d(
                hum_dat.variables["time"][:],
                hum_dat.variables["height"][:],
                hum_dat.variables["absolute_humidity"][:, :],
                tem_dat.variables["time"][:],
                tem_dat.variables["height"][:],
            )

        rpg_dat["height"] = tem_dat.variables["height"][:]
        pres = np.interp(
            tem_dat.variables["time"][:], lev1["time"][:], lev1["air_pressure"][:]
        )
        if data_type == "2P04":
            rpg_dat["relative_humidity"] = rel_hum(
                tem_dat.variables["temperature"][:, :], hum_int
            )
        if data_type == "2P07":
            rpg_dat["potential_temperature"] = pot_tem(
                tem_dat.variables["temperature"][:, :],
                hum_int,
                pres,
                rpg_dat["height"],
            )
        if data_type == "2P08":
            rpg_dat["equivalent_potential_temperature"] = eq_pot_tem(
                tem_dat.variables["temperature"][:, :],
                hum_int,
                pres,
                rpg_dat["height"],
            )

        _combine_lev1(
            tem_dat,
            rpg_dat,
            np.arange(len(tem_dat.variables["time"][:])),
            data_type,
            scan_time,
        )
    return rpg_dat, coeff, index, scan_time


def _get_qf(
    rpg_dat: dict,
    lev1: nc.Dataset,
    coeff: dict,
    index: np.ndarray,
    index_ret: np.ndarray,
    product: str,
) -> None:
    rpg_dat[product + "_quality_flag"] = ma.masked_all((len(index)), np.int32)
    rpg_dat[product + "_quality_flag_status"] = ma.masked_all((len(index)), np.int32)

    _, freq_ind, _ = np.intersect1d(
        lev1["frequency"][:],
        coeff["FR"][:],
        assume_unique=False,
        return_indices=True,
    )
    rpg_dat[product + "_quality_flag"][index_ret] = np.bitwise_or.reduce(
        lev1["quality_flag"][:, freq_ind][index[index_ret]], axis=1
    )
    seqs_all = [
        (key, len(list(val))) for key, val in groupby(lev1["pointing_flag"][:] & 1 > 0)
    ]
    seqs = np.array(
        [
            (key, sum(s[1] for s in seqs_all[:i]), length)
            for i, (key, length) in enumerate(seqs_all)
            if bool(key) is True
        ]
    )

    if product == "temperature" and len(seqs) > 0:
        i_scn = np.where(seqs[:, 2] == int(np.round(np.median(seqs[:, 2]))))[0]
        if len(i_scn) == len(rpg_dat[product + "_quality_flag"][:]):
            for ind, val in enumerate(i_scn):
                scan = np.arange(seqs[val, 1], seqs[val, 1] + seqs[val, 2])
                flg = np.bitwise_or.reduce(lev1["quality_flag"][scan, freq_ind], axis=1)
                rpg_dat[product + "_quality_flag"][ind] = np.bitwise_or.reduce(flg)

    rpg_dat[product + "_quality_flag_status"][index_ret] = lev1["quality_flag_status"][
        :, freq_ind[0]
    ][index[index_ret]]


def _combine_lev1(
    lev1: nc.Dataset,
    rpg_dat: dict,
    index: np.ndarray,
    data_type: str,
    scan_time: np.ndarray,
) -> None:
    """Add level1 data."""
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
    if index.any():
        for ivars in lev1_vars:
            if ivars not in lev1.variables:
                continue
            if ivars == "time_bnds" and data_type == "2P02":
                time = lev1["time"][:][index]
                rpg_dat[ivars] = np.ndarray((len(index), 2))
                rpg_dat[ivars][:, 0] = time - scan_time
                rpg_dat[ivars][:, 1] = time
            elif ivars == "time_bnds" and data_type in ("2P04", "2P07", "2P08"):
                rpg_dat[ivars] = ma.masked_all(lev1[ivars].shape, np.int32)
            else:
                try:
                    rpg_dat[ivars] = lev1[ivars][:][index]
                except IndexError:
                    rpg_dat[ivars] = lev1[ivars][:]


def _del_att(global_attributes: dict) -> None:
    """Remove lev1 only attributes."""
    att_del = ["ir_instrument", "met_instrument", "_accuracy"]
    att_names = global_attributes.keys()
    for name in list(att_names):
        if any(x in name for x in att_del):
            del global_attributes[name]


def load_product(filename: str):
    """Load existing lev2 file for deriving other products."""
    file = nc.Dataset(filename)
    return file


def ele_retrieval(ele_obs: np.ndarray, coeff: dict) -> np.ndarray:
    """Extracts elevation angles used in retrieval."""
    ele_ret = coeff["AG"]
    if ele_ret.shape == ():
        ele_ret = np.array([ele_ret])
    ind = np.argwhere(np.abs(ele_obs - ele_ret[:, np.newaxis]) < 0.5)[:, 0]
    return ele_ret[ind]


def retrieval_input(lev1: dict | nc.Dataset, coeff: dict) -> np.ndarray:
    """Get retrieval input."""
    time_median = ma.median(lev1["time"][:])
    if time_median < 24:
        assert isinstance(lev1, nc.Dataset)
        date = [lev1.year, lev1.month, lev1.day]
        time_median = decimal_hour2unix(date, time_median)

    _, freq_ind, _ = np.intersect1d(
        lev1["frequency"][:],
        coeff["FR"][:],
        assume_unique=False,
        return_indices=True,
    )
    bias = np.ones((len(lev1["time"][:]), 1), np.float32)

    latitude = float(ma.median(lev1["latitude"][:]))
    longitude = float(ma.median(lev1["longitude"][:]))

    if coeff["RT"] == -1:
        ret_in = lev1["tb"][:, :]
    elif coeff["RT"] in (0, 1):
        ret_in = lev1["tb"][:, freq_ind]
    else:
        ret_in = np.concatenate((bias, lev1["tb"][:, freq_ind]), axis=1)

        _data = coeff.get("TS")
        assert _data is not None and len(_data) > 0
        if _data[0] == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["air_temperature"][:], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        _data = coeff.get("HS")
        assert _data is not None and len(_data) > 0
        if _data[0] == 1:
            ret_in = np.concatenate(
                (
                    ret_in,
                    np.reshape(lev1["relative_humidity"][:], (len(lev1["time"][:]), 1)),
                ),
                axis=1,
            )
        _data = coeff.get("PS")
        assert _data is not None and len(_data) > 0
        if _data[0] == 1:
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
            doy = ma.masked_all((len(lev1["time"][:]), 2), np.float32)
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(
                lng=longitude,
                lat=latitude,
            )
            assert timezone_str is not None
            timezone = pytz.timezone(timezone_str)
            dtime = datetime.fromtimestamp(time_median, timezone)
            dyear = datetime(dtime.year, 12, 31, 0, 0).timetuple().tm_yday
            doy[:, 0] = np.cos(
                datetime.fromtimestamp(time_median).timetuple().tm_yday
                / dyear
                * 2
                * np.pi
            )
            doy[:, 1] = np.sin(
                datetime.fromtimestamp(time_median).timetuple().tm_yday
                / dyear
                * 2
                * np.pi
            )
            ret_in = np.concatenate((ret_in, doy), axis=1)
        if coeff.get("SU") == 1:
            sun = ma.masked_all((len(lev1["time"][:]), 2), np.float32)
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(
                lng=longitude,
                lat=latitude,
            )
            assert timezone_str is not None
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


def decimal_hour2unix(date: list, time: np.ndarray) -> np.ndarray | int:
    unix_timestamp = np.datetime64("-".join(date)).astype("datetime64[s]").astype("int")
    return (time * 60 * 60 + unix_timestamp).astype(int)


def _get_retrieval_frequencies(coeff: dict) -> np.ndarray:
    if isinstance(coeff["FR"], ma.MaskedArray):
        frequencies = coeff["FR"][~coeff["FR"][:].mask]
    else:
        frequencies = coeff["FR"]
    return _format_attribute_array(frequencies)


def _combine_array_attributes(tem_dat: dict, hum_dat: dict, name: str) -> np.ndarray:
    a = getattr(tem_dat["temperature"], name)
    b = getattr(hum_dat["absolute_humidity"], name)
    combined = np.hstack((a, b))
    return _format_attribute_array(combined)


def _format_attribute_array(array: np.ndarray | list) -> np.ndarray:
    return np.round(np.sort(np.unique(array)), 2)
