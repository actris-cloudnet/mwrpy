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
            .strftime("%Y-%m-%d")
        )
        today = float(datetime.datetime.today().strftime("%Y"))
        if float(date[0:4]) > today:
            date = (
                datetime.datetime.utcfromtimestamp(
                    utils.epoch2unix(
                        time_median, self.header["_time_ref"], (1970, 1, 1)
                    )
                )
                .strftime("%Y-%m-%d")
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
            if "-".join(utils.seconds2date(t)[:3]) == self.date:
                ind[i] = 1
        self._screen(np.where(ind == 1)[0])

    def _screen(self, ind: np.ndarray):
        if len(ind) < 1:
            raise RuntimeError(["Error: no valid data for date: " + str(self.date)])
        n_time = len(self.data["time"])
        for key, array in self.data.items():
            if isinstance(array, (list, tuple, np.ndarray)):
                if array.shape[0] == n_time:
                    if array.ndim == 1:
                        screened_data = array[ind]
                    else:
                        screened_data = array[ind, :]
                    self.data[key] = screened_data


def read_bls(file_name: str) -> tuple[dict, dict]:
    """This function reads RPG MWR .BLS binary files."""
    version: Literal[2]
    with open(file_name, "rb") as file:
        header = _read_from_file(
            file,
            [("_code", "<i4"), ("n", "<i4"), ("_n_f", "<i4")],
        )
        if header["_code"] == 567846000:
            version = 2
        else:
            raise RuntimeError(f"Error: BLS file code {header['_code']} not supported")
        header |= _read_from_file(
            file,
            [
                ("_xmin", "<f", header["_n_f"]),
                ("_xmax", "<f", header["_n_f"]),
                ("_time_ref", "<i4"),
                ("_f", "<f", header["_n_f"]),
                ("_n_ang", "<i4"),
            ],
        )
        header |= _read_from_file(file, [("_ang", "<f", header["_n_ang"])])

        dt = [
            ("time", "<i4"),
            ("rain", "b"),
            ("temp_sfc", "<f"),
            ("tb", "<f", header["_n_f"]),
            ("_angles", "<i4"),
        ]
        data = _read_from_file(file, dt, header["n"] * header["_n_ang"])
        _check_eof(file)

    data["elevation_angle"], data["azimuth_angle"] = _decode_angles(
        data["_angles"], version
    )
    header = _fix_header(header)
    return header, data


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
            raise RuntimeError(f"Error: BLB file code {header['_code']} not supported")
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
        dt = [("time", "<i4"), ("rain", "b")]
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


FuncType: TypeAlias = Callable[[str], tuple[dict, dict]]
type_reader: dict[str, FuncType] = {
    "brt": read_brt,
    "irt": read_irt,
    "met": read_met,
    "hkd": read_hkd,
    "blb": read_blb,
    "bls": read_bls,
}
