"""This module contains all functions to read in RPG MWR binary files"""
import datetime
import logging
from collections.abc import Callable
from io import SEEK_END
from typing import BinaryIO, Literal, TypeAlias

import numpy as np

from mwrpy import utils

Fill_Value_Float = -999.0
Fill_Value_Int = -99


def stack_files(file_list: list[str]) -> tuple[dict, dict]:
    """This function calls extension specific reader and stacks data and header."""

    def _stack_data(source: dict, target: dict, fun: Callable):
        for name, value in source.items():
            target[name] = fun((target[name], value)) if name in target else value

    def _stack_header(source: dict, target: dict, fun: Callable):
        for name, value in source.items():
            if not name.startswith("_"):
                target[name] = fun(target[name], value) if name in target else value
            else:
                target[name] = value

    ext = str(file_list[0][-3:]).lower()
    if ext not in (
        "brt",
        "irt",
        "met",
        "hkd",
        "blb",
        "bls",
        "spc",
        "his",
        "iwv",
        "tpb",
        "tpc",
    ):
        raise RuntimeError(["Error: no reader for file type " + ext])

    read_type = type_reader[ext]
    data: dict = {}
    header: dict = {}

    for file in file_list:
        try:
            header_tmp, data_tmp = read_type(file)
        except (TypeError, ValueError) as err:
            logging.warning(err)
            continue
        _stack_header(header_tmp, header, np.add)
        _stack_data(data_tmp, data, np.concatenate)

    return header, data


class RpgBin:
    """Class for RPG binary files"""

    def __init__(self, file_list: list[str]):
        self.header, self.raw_data = stack_files(file_list)
        self.raw_data["time"] = utils.epoch2unix(
            self.raw_data["time"], self.header["_time_ref"]
        )
        self.date = self._get_date()
        self.data: dict = {}
        self._init_data()
        if str(file_list[0][-3:]).lower() != "his":
            self.find_valid_times()

    def _init_data(self):
        for key, data in self.raw_data.items():
            self.data[key] = data

    def _get_date(self):
        time_median = float(np.ma.median(self.raw_data["time"]))
        date = (
            datetime.datetime.utcfromtimestamp(
                utils.epoch2unix(time_median, self.header["_time_ref"])
            )
            .strftime("%Y %m %d")
            .split()
        )
        today = float(datetime.datetime.today().strftime("%Y"))
        if float(date[0]) > today:
            date = (
                datetime.datetime.utcfromtimestamp(
                    utils.epoch2unix(
                        time_median, self.header["_time_ref"], (1970, 1, 1)
                    )
                )
                .strftime("%Y %m %d")
                .split()
            )
        return date

    def find_valid_times(self):
        # sort timestamps
        time = self.data["time"]
        ind = time.argsort()
        self._screen(ind)

        # remove duplicate timestamps
        time = self.data["time"]
        _, ind = np.unique(time, return_index=True)
        self._screen(ind)

        # find valid date
        time = self.data["time"]
        ind = np.zeros(len(time), dtype=np.int32)
        for i, t in enumerate(time):
            if utils.seconds2date(t)[:3] == self.date:
                ind[i] = 1
        self._screen(np.where(ind == 1)[0])

    def _screen(self, ind: np.ndarray):
        if len(ind) < 1:
            raise RuntimeError(["Error: no valid data for date: " + self.date])
        n_time = len(self.data["time"])
        for key, array in self.data.items():
            data = array
            if data.ndim > 0 and data.shape[0] == n_time:
                if data.ndim == 1:
                    screened_data = data[ind]
                else:
                    screened_data = data[ind, :]
                self.data[key] = screened_data


def read_brt(file_name: str) -> tuple[dict, dict]:
    """Reads BRT files and returns header and data as dictionary."""
    version: Literal[1, 2]
    with open(file_name, "rb") as file:
        header = _read_from_file(
            file,
            [("_code", "<i4"), ("n", "<i4"), ("_time_ref", "<i4"), ("_n_f", "<i4")],
        )
        if header["_code"] == 666666:
            version = 1
        elif header["_code"] == 666000:
            version = 2
        else:
            raise RuntimeError(f"Error: BRT file code {header['_code']} not supported")
        header |= _read_from_file(
            file,
            [
                ("_f", "<f", header["_n_f"]),
                ("_xmin", "<f", header["_n_f"]),
                ("_xmax", "<f", header["_n_f"]),
            ],
        )
        data = _read_from_file(
            file,
            [
                ("time", "<i4"),
                ("rain", "b"),
                ("tb", "<f", header["_n_f"]),
                ("_angles", "<f" if version == 1 else "<i4"),
            ],
            header["n"],
        )
        _check_eof(file)

    ele, azi = _decode_angles(data["_angles"], version)
    data["elevation_angle"], data["azimuth_angle"] = ele, azi
    header = _fix_header(header)
    return header, data


def read_blb(file_name: str) -> tuple[dict, dict]:
    """Reads BLB files and returns header and data as dictionary."""
    with open(file_name, "rb") as file:
        header = _read_from_file(file, [("_code", "<i4"), ("n", "<i4")])
        if header["_code"] == 567845847:
            header["_n_f"] = 14
            version = 1
        elif header["_code"] == 567845848:
            header |= _read_from_file(file, [("_n_f", "<i4")])
            version = 2
        else:
            raise RuntimeError(f"Error: BRT file code {header['_code']} not supported")
        header |= _read_from_file(
            file,
            [
                ("_xmin", "<f", header["_n_f"]),
                ("_xmax", "<f", header["_n_f"]),
                ("_time_ref", "<i4"),
            ],
        )
        if version == 1:
            header |= _read_from_file(file, [("_n_f", "<i4")])
        header |= _read_from_file(
            file, [("_f", "<f", header["_n_f"]), ("_n_ang", "<i4")]
        )
        header |= _read_from_file(file, [("_ang", "<f", header["_n_ang"])])
        dt = [("time", "<i4"), ("rf_mod", "b")]
        for n in range(header["_n_f"]):
            dt += [(f"tb_{n}", "<f", header["_n_ang"])]
            dt += [(f"temp_sfc_{n}", "<f")]
        data = _read_from_file(file, dt, header["n"])
        _check_eof(file)

    data_out = {
        "tb": np.empty((header["n"], header["_n_f"], header["_n_ang"])),
        "temp_sfc": np.empty((header["n"], header["_n_f"])),
    }
    for key in data:
        if "tb_" in key:
            freq_ind = int(key.split("_")[-1])
            data_out["tb"][:, int(freq_ind)] = data[key]
        elif "temp_sfc_" in key:
            freq_ind = int(key.split("_")[-1])
            data_out["temp_sfc"][:, freq_ind] = data[key]
        else:
            data_out[key] = data[key]

    header["_ang"] = np.flip(header["_ang"])  # Flipped in the original code
    header = _fix_header(header)
    return header, data_out


def read_irt(file_name: str) -> tuple[dict, dict]:
    """Reads IRT files and returns header and data as dictionary."""
    version: Literal[1, 2, 3]

    with open(file_name, "rb") as file:
        header = _read_from_file(
            file,
            [
                ("_code", "<i4"),
                ("n", "<i4"),
                ("_xmin", "<f"),
                ("_xmax", "<f"),
                ("_time_ref", "<i4"),
            ],
        )
        match header["_code"]:
            case 671112495:
                version = 1
            case 671112496:
                version = 2
            case 671112000:
                version = 3
            case _:
                raise ValueError(
                    f"Error: IRT file code {header['_code']} not supported"
                )
        if version == 1:
            header["_n_f"] = 1
            header["_f"] = 11.1
        else:
            header |= _read_from_file(file, [("_n_f", "<i4")])
            header |= _read_from_file(file, [("_f", "<f", (header["_n_f"],))])
        dt = [("time", "<i4"), ("rain", "b"), ("irt", "<f", (header["_n_f"],))]
        if version > 1:
            dt += [("_angles", "<f" if version == 2 else "<i4")]
        data = _read_from_file(file, dt, header["n"])
        _check_eof(file)

    if "_angles" in data:
        ele, azi = _decode_angles(data["_angles"], 1 if version == 2 else 2)
        data["ir_elevation_angle"], data["ir_azimuth_angle"] = ele, azi
    data["irt"] += 273.15

    header = _fix_header(header)
    return header, data


def read_hkd(file_name: str) -> tuple[dict, dict]:
    """Reads HKD files and returns header and data as dictionary."""
    with open(file_name, "rb") as file:
        header = _read_from_file(
            file,
            [("_code", "<i4"), ("n", "<i4"), ("_time_ref", "<i4"), ("_sel", "<i4")],
        )
        dt = [("time", "<i4"), ("alarm", "b")]
        if header["_sel"] & 0x1:
            dt += [("longitude", "<f"), ("latitude", "<f")]
        if header["_sel"] & 0x2:
            dt += [("temp", "<f", 4)]
        if header["_sel"] & 0x4:
            dt += [("stab", "<f", 2)]
        if header["_sel"] & 0x8:
            dt += [("flash", "<i4")]
        if header["_sel"] & 0x10:
            dt += [("qual", "<i4")]
        if header["_sel"] & 0x20:
            dt += [("status", "<i4")]
        data = _read_from_file(file, dt, header["n"])
        _check_eof(file)

    header = _fix_header(header)
    return header, data


def read_met(file_name: str) -> tuple[dict, dict]:
    """Reads MET files and returns header and data as dictionary."""
    with open(file_name, "rb") as file:
        header = _read_from_file(file, [("_code", "<i4"), ("n", "<i4")])
        if header["_code"] == 599658943:
            header["_n_add"] = 0
        elif header["_code"] == 599658944:
            header |= _read_from_file(file, [("_n_add", "b")])
        else:
            raise ValueError(f"Error: MET file code {header['_code']} not supported")
        dt = [
            ("time", "<i4"),
            ("rain", "b"),
            ("air_pressure", "<f"),
            ("air_temperature", "<f"),
            ("relative_humidity", "<f"),
        ]
        hdt = [
            ("_air_pressure_min", "<f"),
            ("_air_pressure_max", "<f"),
            ("_air_temperature_min", "<f"),
            ("_air_temperature_max", "<f"),
            ("_relative_humidity_min", "<f"),
            ("_relative_humidity_max", "<f"),
        ]
        if header["_n_add"] & 0x1:
            dt.append(("wind_speed", "<f"))
            hdt.append(("_wind_speed_min", "<f"))
            hdt.append(("_wind_speed_max", "<f"))
        if header["_n_add"] & 0x2:
            dt.append(("wind_direction", "<f"))
            hdt.append(("_wind_direction_min", "<f"))
            hdt.append(("_wind_direction_max", "<f"))
        if header["_n_add"] & 0x4:
            dt.append(("rainfall_rate", "<f"))
            hdt.append(("_rainfall_rate_min", "<f"))
            hdt.append(("_rainfall_rate_max", "<f"))
        if header["_n_add"] & 0x8:
            dt.append(("_adds4", "<f"))
            hdt.append(("_adds4_min", "<f"))
            hdt.append(("_adds4_max", "<f"))
        if header["_n_add"] & 0x10:
            dt.append(("_adds5", "<f"))
            hdt.append(("_adds5_min", "<f"))
            hdt.append(("_adds5_max", "<f"))
        if header["_n_add"] & 0x20:
            dt.append(("_adds6", "<f"))
            hdt.append(("_adds6_min", "<f"))
            hdt.append(("_adds6_max", "<f"))
        if header["_n_add"] & 0x40:
            dt.append(("_adds7", "<f"))
            hdt.append(("_adds7_min", "<f"))
            hdt.append(("_adds7_max", "<f"))
        if header["_n_add"] & 0x80:
            dt.append(("_adds8", "<f"))
            hdt.append(("_adds8_min", "<f"))
            hdt.append(("_adds8_max", "<f"))
        hdt.append(("_time_ref", "<i4"))
        header |= _read_from_file(file, hdt)
        data = _read_from_file(file, dt, header["n"])
        _check_eof(file)

    data["relative_humidity"] /= 100  # Converted in the original code
    header = _fix_header(header)
    return header, data


Dim = int | tuple[int, ...]
Field = tuple[str, str] | tuple[str, str, Dim]


def _read_from_file(file: BinaryIO, fields: list[Field], count: int = 1) -> dict:
    arr = np.fromfile(file, np.dtype(fields), count)
    if (read := len(arr)) != count:
        raise IOError(f"Read {read} of {count} records from file")
    if count == 1:
        arr = arr[0]
    return {field: arr[field] for field, *args in fields}


def _check_eof(file: BinaryIO):
    current_offset = file.tell()
    file.seek(0, SEEK_END)
    end_offset = file.tell()
    if current_offset != end_offset:
        raise IOError(f"{end_offset - current_offset} unread bytes")


def _decode_angles(
    x: np.ndarray, method: Literal[1, 2]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode elevation and azimuth angles.
    >>> _decode_angles(np.array([1267438.5]), method=1)
    (array([138.5]), array([267.4]))
    >>> _decode_angles(np.array([1453031045, -900001232]), method=2)
    (array([145.3, -90. ]), array([310.45,  12.32]))
    Based on `interpret_angle` from mwr_raw2l1 licensed under BSD 3-Clause:
    https://github.com/MeteoSwiss/mwr_raw2l1/blob/0738490d22f77138cdf9329bf102f319c78be584/mwr_raw2l1/readers/reader_rpg_helpers.py#L30
    """

    if method == 1:
        # Description in the manual is quite unclear so here's an improved one:
        # Ang=sign(El)*(|El|+1000*Az), -90°<=El<100°, 0°<=Az<360°. If El>=100°
        # (i.e. requires 3 digits), the value 1000.000 is added to Ang and El in
        # the formula is El-100°. For decoding to make sense, Az and El must be
        # stored in precision of 0.1 degrees.

        ele_offset = np.zeros(x.shape)
        ind_offset_corr = x >= 1e6
        ele_offset[ind_offset_corr] = 100
        x = np.copy(x)
        x[ind_offset_corr] -= 1e6

        azi = (np.abs(x) // 100) / 10
        ele = x - np.sign(x) * azi * 1000 + ele_offset
    else:
        # First 5 decimal digits is azimuth*100, last 5 decimal digits is
        # elevation*100, sign of Ang is sign of elevation.
        ele = np.sign(x) * (np.abs(x) // 1e5) / 100
        azi = (np.abs(x) - np.abs(ele) * 1e7) / 100

    return ele, azi


def _fix_header(header: dict) -> dict:
    """Maybe can get rid of this function later."""
    for key in header:
        if key in ["_code", "_time_ref", "_sel", "_n_add"]:
            header[key] = np.array(header[key], dtype=int)
        elif key in ["_xmin", "_xmax", "_f", "_ang"]:
            header[key] = np.array(header[key])
        elif key in ["n", "_n_f", "_n_ang"]:
            header[key] = int(header[key])
    return header


# Refactor these when they are needed:


def read_tpb(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .TPB binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code != 459769847:
            raise RuntimeError(["Error: TPB file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            xmin = np.fromfile(file, np.float32, 1)
            xmax = np.fromfile(file, np.float32, 1)
            time_ref = np.fromfile(file, np.uint32, 1)
            ret_type = np.fromfile(file, np.uint32, 1)
            alt_anz = int(np.fromfile(file, np.uint32, 1))
            alts = np.fromfile(file, np.uint32, alt_anz)

            header_names = [
                "_code",
                "n",
                "_xmin",
                "_xmax",
                "_time_ref",
                "_ret_type",
                "_alt_anz",
                "_alts",
            ]
            header_values = [code, n, xmin, xmax, time_ref, ret_type, alt_anz, alts]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rain": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "T": np.ones((header["n"], header["_alt_anz"]), np.float32)
                * Fill_Value_Float,
            }
            return vrs

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["T"][sample,] = np.fromfile(file, np.float32, header["_alt_anz"])
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_iwv(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .IWV binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code not in (594811068, 594811000):
            raise RuntimeError(["Error: IWV file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            xmin = np.fromfile(file, np.float32, 1)
            xmax = np.fromfile(file, np.float32, 1)
            time_ref = np.fromfile(file, np.uint32, 1)
            ret_type = np.fromfile(file, np.uint32, 1)

            header_names = ["_code", "n", "_xmin", "_xmax", "_time_ref", "_ret_type"]
            header_values = [code, n, xmin, xmax, time_ref, ret_type]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rain": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "iwv": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "iwv_ele": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "iwv_azi": np.ones(header["n"], np.float32) * Fill_Value_Float,
            }
            return vrs

        def _angle_calc(ang, code):
            """Convert angle"""

            if code == 594811068:
                els = ang - 100.0 * ((ang / 100.0).astype(np.int32))
                azs = (ang - els) / 1000.0
                if azs <= 360.0:
                    el = els
                    az = azs
                elif azs > 1000.0:
                    az = azs - 1000.0
                    el = 100.0 + els
            elif code == 594811000:
                a_str = str(ang[0])
                if a_str[0:-5].isnumeric():
                    el = float(a_str[0:-5]) / 100.0
                else:
                    el = Fill_Value_Float
                if a_str[-5:].isnumeric():
                    az = float(a_str[-5:]) / 100.0
                else:
                    az = Fill_Value_Float

            return el, az

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["iwv"][sample] = np.fromfile(file, np.float32, 1)
                if code == 594811068:
                    ang = np.fromfile(file, np.float32, 1)
                elif code == 594811000:
                    ang = np.fromfile(file, np.int32, 1)
                data["iwv_ele"][sample], data["iwv_azi"][sample] = _angle_calc(
                    ang, code
                )
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_bls(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .BLS binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code != 567846000:
            raise RuntimeError(["Error: BLS file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            n_f = int(np.fromfile(file, np.int32, 1))
            xmin = np.fromfile(file, np.float32, n_f)
            xmax = np.fromfile(file, np.float32, n_f)
            time_ref = np.fromfile(file, np.uint32, 1)
            f = np.fromfile(file, np.float32, n_f)
            n_ang = int(np.fromfile(file, np.int32, 1))
            ang = np.fromfile(file, np.float32, n_ang)

            header_names = [
                "_code",
                "n",
                "_n_f",
                "_xmin",
                "_xmax",
                "_time_ref",
                "_f",
                "_n_ang",
                "_ang",
            ]
            header_values = [code, n, n_f, xmin, xmax, time_ref, f, n_ang, ang]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"] * header["_n_ang"], np.int32)
                * Fill_Value_Int,
                "rain": np.ones(header["n"] * header["_n_ang"], np.byte)
                * Fill_Value_Int,
                "temp_sfc": np.ones(header["n"] * header["_n_ang"], np.float32)
                * Fill_Value_Float,
                "tb": np.ones(
                    [header["n"] * header["_n_ang"], header["_n_f"]], np.float32
                )
                * Fill_Value_Float,
                "elevation_angle": np.ones(header["n"] * header["_n_ang"], np.float32)
                * Fill_Value_Float,
                "azimuth_angle": np.ones(header["n"] * header["_n_ang"], np.float32)
                * Fill_Value_Float,
            }
            return vrs

        def _angle_calc(ang):
            """Convert angle"""

            a_str = str(ang[0])
            if a_str[0:-5].isnumeric():
                el = float(a_str[0:-5]) / 100.0
            else:
                el = Fill_Value_Float
            if a_str[-5:].isnumeric():
                az = float(a_str[-5:]) / 100.0
            else:
                az = Fill_Value_Float

            return el, az

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"] * header["_n_ang"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["temp_sfc"][sample] = np.fromfile(file, np.float32, 1)
                data["tb"][sample,] = np.fromfile(file, np.float32, header["_n_f"])
                ang = np.fromfile(file, np.int32, 1)
                (
                    data["elevation_angle"][sample],
                    data["azimuth_angle"][sample],
                ) = _angle_calc(ang)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_spc(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .SPC binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code not in (666667, 667000):
            raise RuntimeError(["Error: SPC file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            time_ref = np.fromfile(file, np.uint32, 1)
            n_f = int(np.fromfile(file, np.int32, 1))
            f = np.fromfile(file, np.float32, n_f)
            xmin = np.fromfile(file, np.float32, n_f)
            xmax = np.fromfile(file, np.float32, n_f)

            header_names = ["_code", "n", "_time_ref", "_n_f", "_f", "_xmin", "_xmax"]
            header_values = [code, n, time_ref, n_f, f, xmin, xmax]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rain": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "tb": np.ones([header["n"], header["_n_f"]], np.float32)
                * Fill_Value_Float,
                "elevation_angle": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "azimuth_angle": np.ones(header["n"], np.float32) * Fill_Value_Float,
            }
            return vrs

        def _angle_calc(ang, code):
            """Convert angle"""

            if code == 666667:
                sign = 1
                if ang < 0:
                    sign = -1
                az = sign * ((ang / 100.0).astype(np.int32)) / 10.0
                el = ang - (sign * az * 1000.0)

            elif code == 667000:
                a_str = str(ang[0])
                el = float(a_str[0:-5]) / 100.0
                az = float(a_str[-5:]) / 100.0
            return el, az

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["tb"][sample,] = np.fromfile(file, np.float32, header["_n_f"])
                if code == 666667:
                    ang = np.fromfile(file, np.float32, 1)
                elif code == 667000:
                    ang = np.fromfile(file, np.int32, 1)
                (
                    data["elevation_angle"][sample],
                    data["azimuth_angle"][sample],
                ) = _angle_calc(ang, code)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_his(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR ABSCAL.HIS binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code != 39583209:
            raise RuntimeError(["Error: CAL file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            time_ref = 1
            header_names = ["_code", "n", "_time_ref"]
            header_values = [code, n, time_ref]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "len": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rad_id": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "cal1_t": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "cal2_t": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "t1": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "t2": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "a_temp1": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "a_temp2": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "p1": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "p2": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "hl_temp1": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "hl_temp2": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "cl_temp1": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "cl_temp2": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "spare": np.ones([header["n"], 5], np.float32) * Fill_Value_Float,
                "n_ch1": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "freq1": np.ones([header["n"], 7], np.float32) * Fill_Value_Float,
                "n_ch2": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "freq2": np.ones([header["n"], 7], np.float32) * Fill_Value_Float,
                "cal_flag": np.ones([header["n"], 14], np.int32) * Fill_Value_Int,
                "gain": np.ones([header["n"], 14], np.float32) * Fill_Value_Float,
                "tn": np.ones([header["n"], 14], np.float32) * Fill_Value_Float,
                "t_sys": np.ones([header["n"], 14], np.float32) * Fill_Value_Float,
                "alpha": np.ones([header["n"], 14], np.float32) * Fill_Value_Float,
            }
            return vrs

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["len"][sample] = np.fromfile(file, np.int32, 1)
                data["rad_id"][sample] = np.fromfile(file, np.int32, 1)
                data["cal1_t"][sample] = np.fromfile(file, np.int32, 1)
                data["cal2_t"][sample] = np.fromfile(file, np.int32, 1)
                data["t1"][sample] = np.fromfile(file, np.int32, 1)
                data["time"][sample] = data["t1"][sample]
                data["t2"][sample] = np.fromfile(file, np.int32, 1)
                data["a_temp1"][sample] = np.fromfile(file, np.float32, 1)
                data["a_temp2"][sample] = np.fromfile(file, np.float32, 1)
                data["p1"][sample] = np.fromfile(file, np.float32, 1)
                data["p2"][sample] = np.fromfile(file, np.float32, 1)
                data["hl_temp1"][sample] = np.fromfile(file, np.float32, 1)
                data["hl_temp2"][sample] = np.fromfile(file, np.float32, 1)
                data["cl_temp1"][sample] = np.fromfile(file, np.float32, 1)
                data["cl_temp2"][sample] = np.fromfile(file, np.float32, 1)
                data["spare"][sample,] = np.fromfile(file, np.float32, 5)
                data["n_ch1"][sample] = np.fromfile(file, np.int32, 1)
                data["n_ch1"][sample] = data["n_ch1"][sample]
                data["freq1"][sample, 0 : data["n_ch1"][sample]] = np.fromfile(
                    file, np.float32, int(data["n_ch1"][sample])
                )
                data["n_ch2"][sample] = np.fromfile(file, np.int32, 1)
                data["freq2"][sample, 0 : int(data["n_ch2"][sample])] = np.fromfile(
                    file, np.float32, int(data["n_ch2"][sample])
                )
                data["cal_flag"][
                    sample, 0 : int(data["n_ch1"][sample] + data["n_ch2"][sample])
                ] = np.fromfile(
                    file, np.int32, int(data["n_ch1"][sample] + data["n_ch2"][sample])
                )
                data["gain"][
                    sample, 0 : int(data["n_ch1"][sample] + data["n_ch2"][sample])
                ] = np.fromfile(
                    file, np.float32, int(data["n_ch1"][sample] + data["n_ch2"][sample])
                )
                data["tn"][
                    sample, 0 : int(data["n_ch1"][sample] + data["n_ch2"][sample])
                ] = np.fromfile(
                    file, np.float32, int(data["n_ch1"][sample] + data["n_ch2"][sample])
                )
                data["t_sys"][
                    sample, 0 : int(data["n_ch1"][sample] + data["n_ch2"][sample])
                ] = np.fromfile(
                    file, np.float32, int(data["n_ch1"][sample] + data["n_ch2"][sample])
                )
                data["alpha"][
                    sample, 0 : int(data["n_ch1"][sample] + data["n_ch2"][sample])
                ] = np.fromfile(
                    file, np.float32, int(data["n_ch1"][sample] + data["n_ch2"][sample])
                )

            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


FuncType: TypeAlias = Callable[[str], tuple[dict, dict]]
type_reader: dict[str, FuncType] = {
    "brt": read_brt,
    "irt": read_irt,
    "met": read_met,
    "hkd": read_hkd,
    "blb": read_blb,
    "bls": read_bls,
    "spc": read_spc,
    "his": read_his,
    "iwv": read_iwv,
    "tpb": read_tpb,
}


# Just for testing, delete these later:


def read_blb_legacy(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .BLB binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code not in (567845847, 567845848):
            raise RuntimeError(["Error: BLB file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            if code == 567845848:
                n_f = int(np.fromfile(file, np.int32, 1))
            else:
                n_f = 14
            xmin = np.fromfile(file, np.float32, n_f)
            xmax = np.fromfile(file, np.float32, n_f)
            time_ref = np.fromfile(file, np.uint32, 1)
            if code == 567845847:
                n_f = int(np.fromfile(file, np.int32, 1))
            f = np.fromfile(file, np.float32, n_f)
            n_ang = int(np.fromfile(file, np.int32, 1))
            ang = np.flip(np.fromfile(file, np.float32, n_ang))
            if ang[0] > 1000.0:
                for ind, val in enumerate(ang):
                    sign = 1
                    if val < 0:
                        sign = -1
                    az = sign * ((val / 100.0).astype(np.int32)) / 10.0
                    ang[ind] = val - (sign * az * 1000.0)

            header_names = [
                "_code",
                "n",
                "_xmin",
                "_xmax",
                "_time_ref",
                "_n_f",
                "_f",
                "_n_ang",
                "_ang",
            ]
            header_values = [code, n, xmin, xmax, time_ref, n_f, f, n_ang, ang]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rf_mod": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "temp_sfc": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "tb": np.ones(
                    [header["n"], header["_n_f"], header["_n_ang"]], np.float32
                )
                * Fill_Value_Float,
            }
            return vrs

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rf_mod"][sample] = np.fromfile(file, np.byte, 1)
                for freq in range(header["_n_f"]):
                    data["tb"][
                        sample,
                        freq,
                    ] = np.fromfile(file, np.float32, header["_n_ang"])
                    data["temp_sfc"][sample] = np.fromfile(file, np.float32, 1)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_brt_legacy(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .BRT binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code not in (666000, 666666):
            raise RuntimeError(["Error: BRT file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            time_ref = np.fromfile(file, np.uint32, 1)
            n_f = int(np.fromfile(file, np.int32, 1))
            f = np.fromfile(file, np.float32, n_f)
            xmin = np.fromfile(file, np.float32, n_f)
            xmax = np.fromfile(file, np.float32, n_f)

            header_names = ["_code", "n", "_time_ref", "_n_f", "_f", "_xmin", "_xmax"]
            header_values = [code, n, time_ref, n_f, f, xmin, xmax]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rain": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "tb": np.ones([header["n"], header["_n_f"]], np.float32)
                * Fill_Value_Float,
                "elevation_angle": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "azimuth_angle": np.ones(header["n"], np.float32) * Fill_Value_Float,
            }
            return vrs

        def _angle_calc(ang, code):
            """Convert angle"""

            if code == 666666:
                sign = 1
                if ang < 0:
                    sign = -1
                az = sign * ((ang / 100.0).astype(np.int32)) / 10.0
                el = ang - (sign * az * 1000.0)

            elif code == 666000:
                a_str = str(ang[0])
                if a_str[0:-5].isnumeric():
                    el = float(a_str[0:-5]) / 100.0
                else:
                    el = Fill_Value_Float
                if a_str[-5:].isnumeric():
                    az = float(a_str[-5:]) / 100.0
                else:
                    az = Fill_Value_Float

            return el, az

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["tb"][sample,] = np.fromfile(file, np.float32, header["_n_f"])
                if code == 666666:
                    ang = np.fromfile(file, np.float32, 1)
                elif code == 666000:
                    ang = np.fromfile(file, np.int32, 1)
                (
                    data["elevation_angle"][sample],
                    data["azimuth_angle"][sample],
                ) = _angle_calc(ang, code)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_irt_legacy(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .IRT binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code not in (671112495, 671112496, 671112000):
            raise RuntimeError(["Error: IRT file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            xmin = np.fromfile(file, np.float32, 1)
            xmax = np.fromfile(file, np.float32, 1)
            time_ref = np.fromfile(file, np.uint32, 1)
            if code == 671112495:
                n_f = 1
                f = 11.1
            else:
                n_f = int(np.fromfile(file, np.uint32, 1))
                f = np.fromfile(file, np.float32, n_f)

            header_names = ["_code", "n", "_xmin", "_xmax", "_time_ref", "_n_f", "_f"]
            header_values = [code, n, xmin, xmax, time_ref, n_f, f]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rain": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "irt": np.ones([header["n"], header["_n_f"]], np.float32)
                * Fill_Value_Float,
                "ir_elevation_angle": np.ones(header["n"], np.float32)
                * Fill_Value_Float,
                "ir_azimuth_angle": np.ones(header["n"], np.float32) * Fill_Value_Float,
            }
            return vrs

        def _angle_calc(ang, code):
            """Convert angle"""
            if code == 671112496:
                els = ang - 100.0 * ((ang / 100.0).astype(np.int32))
                azs = (ang - els) / 1000.0
                if azs <= 360.0:
                    el = els
                    az = azs
                elif azs > 1000.0:
                    az = azs - 1000.0
                    el = 100.0 + els
            elif code == 671112000:
                a_str = str(ang[0])
                if a_str[0:-5].isnumeric():
                    el = float(a_str[0:-5]) / 100.0
                else:
                    el = Fill_Value_Float
                if a_str[-5:].isnumeric():
                    az = float(a_str[-5:]) / 100.0
                else:
                    az = Fill_Value_Float

            return el, az

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["irt"][sample,] = (
                    np.fromfile(file, np.float32, header["_n_f"]) + 273.15
                )
                if code == 671112496:
                    ang = np.fromfile(file, np.float32, 1)
                elif code == 671112000:
                    ang = np.fromfile(file, np.int32, 1)
                (
                    data["ir_elevation_angle"][sample],
                    data["ir_azimuth_angle"][sample],
                ) = _angle_calc(ang, code)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_hkd_legacy(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .HKD binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code != 837854832:
            raise RuntimeError(["Error: HKD file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            time_ref = np.fromfile(file, np.uint32, 1)
            sel = np.fromfile(file, np.uint32, 1)

            header_names = ["_code", "n", "_time_ref", "_sel"]
            header_values = [code, n, time_ref, sel]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "alarm": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "longitude": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "latitude": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "temp": np.ones([header["n"], 4], np.float32) * Fill_Value_Float,
                "stab": np.ones([header["n"], 2], np.float32) * Fill_Value_Float,
                "flash": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "qual": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "status": np.ones(header["n"], np.int32) * Fill_Value_Int,
            }
            return vrs

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["alarm"][sample] = np.fromfile(file, np.byte, count=1)
                if header["_sel"] & 1:
                    data["longitude"][sample] = np.fromfile(file, np.float32, count=1)
                    data["latitude"][sample] = np.fromfile(file, np.float32, count=1)
                if header["_sel"] & 2:
                    data["temp"][sample,] = np.fromfile(file, np.float32, count=4)
                if header["_sel"] & 4:
                    data["stab"][sample,] = np.fromfile(file, np.float32, count=2)
                if header["_sel"] & 8:
                    data["flash"][sample] = np.fromfile(file, np.int32, count=1)
                if header["_sel"] & 16:
                    data["qual"][sample] = np.fromfile(file, np.int32, count=1)
                if header["_sel"] & 32:
                    data["status"][sample] = np.fromfile(file, np.int32, count=1)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data


def read_met_legacy(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .MET binary files."""

    with open(file_name, "rb") as file:
        code = np.fromfile(file, np.int32, 1)
        if code not in (599658943, 599658944):
            raise RuntimeError(["Error: MET file code " + str(code) + " not suported"])

        def _get_header():
            """Read header info"""

            n = int(np.fromfile(file, np.uint32, 1))
            n_add = 0
            if code == 599658944:
                n_add = np.fromfile(file, np.byte, 1)
            n_sen = bin(int(n_add))
            xmin = np.ones(3 + n_sen.count("1"), np.float32) * Fill_Value_Float
            xmax = np.ones(3 + n_sen.count("1"), np.float32) * Fill_Value_Float
            for index in range(3 + n_sen.count("1")):
                xmin[index] = np.fromfile(file, np.float32, 1)
                xmax[index] = np.fromfile(file, np.float32, 1)
            time_ref = np.fromfile(file, np.uint32, 1)

            header_names = [
                "_code",
                "n",
                "_n_add",
                "_n_sen",
                "_xmin",
                "_xmax",
                "_time_ref",
            ]
            header_values = [code, n, n_add, n_sen, xmin, xmax, time_ref]
            header = dict(zip(header_names, header_values))
            return header

        def _create_variables():
            """Initialize data arrays"""

            vrs = {
                "time": np.ones(header["n"], np.int32) * Fill_Value_Int,
                "rain": np.ones(header["n"], np.byte) * Fill_Value_Int,
                "air_pressure": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "air_temperature": np.ones(header["n"], np.float32) * Fill_Value_Float,
                "relative_humidity": np.ones(header["n"], np.float32)
                * Fill_Value_Float,
                "adds": np.ones([header["n"], 3], np.float32) * Fill_Value_Float,
            }
            return vrs

        def _get_data():
            """Loop over file to read data"""

            data = _create_variables()
            for sample in range(header["n"]):
                data["time"][sample] = np.fromfile(file, np.int32, 1)
                data["rain"][sample] = np.fromfile(file, np.byte, 1)
                data["air_pressure"][sample] = np.fromfile(file, np.float32, 1)
                data["air_temperature"][sample] = np.fromfile(file, np.float32, 1)
                data["relative_humidity"][sample] = (
                    np.fromfile(file, np.float32, 1) / 100.0
                )
                for add in range(header["_n_sen"].count("1")):
                    data["adds"][sample, add] = np.fromfile(file, np.float32, 1)
            file.close()
            return data

        header = _get_header()
        data = _get_data()
        return header, data
