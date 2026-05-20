import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mwrpy.level1 import rpg_bin

dir_name = os.path.dirname(os.path.realpath(__file__))


class TestBrtFileReading:
    header, data = rpg_bin.read_brt(f"{dir_name}/data/hyytiala/230406.BRT")

    def test_header(self):
        assert isinstance(self.header, dict)
        expected_header_keys = {
            "_code",
            "n",
            "_time_ref",
            "_n_f",
            "_f",
            "_xmin",
            "_xmax",
        }
        assert set(self.header.keys()) == expected_header_keys
        data = {
            "_code": np.array([666000]),
            "n": 73317,
            "_time_ref": np.array([1]),
            "_n_f": 14,
            "_f": np.array(
                [
                    22.24,
                    23.04,
                    23.84,
                    25.44,
                    26.24,
                    27.84,
                    31.4,
                    51.26,
                    52.28,
                    53.86,
                    54.94,
                    56.66,
                    57.3,
                    58.0,
                ]
            ),
        }
        for key, value in data.items():
            assert isinstance(self.header[key], type(value))
            if isinstance(value, int):
                assert self.header[key] == value
            else:
                assert_array_almost_equal(self.header[key], np.array(value), decimal=2)

    def test_data(self):
        assert isinstance(self.data, dict)
        expected_data_keys = {
            "time",
            "rain",
            "tb",
            "_angles",
            "elevation_angle",
            "azimuth_angle",
        }
        assert set(self.data.keys()) == expected_data_keys
        assert len(self.data["time"]) == self.header["n"]
        assert self.data["time"][0] == 702432051
        assert self.data["time"][-1] == 702518389
        assert np.isclose(self.data["azimuth_angle"], 0.02).all()
        assert np.isclose(self.data["elevation_angle"], 90.0, atol=0.01).all()
        assert np.isclose(self.data["rain"], 0).all()
        assert self.data["tb"].shape == (self.header["n"], self.header["_n_f"])
        assert np.min(self.data["tb"]) > 13
        assert np.max(self.data["tb"]) < 281
        assert np.isclose(np.mean(self.data["tb"]), 124.37, atol=1)


class TestBlbFileReading:
    header, data = rpg_bin.read_blb(f"{dir_name}/data/hyytiala/230406.BLB")

    def test_header(self):
        assert isinstance(self.header, dict)
        data = {
            "_code": np.array([567845848]),
            "n": 144,
            "_xmin": np.array(
                [
                    23.304903,
                    22.66079,
                    19.596289,
                    15.603367,
                    14.533482,
                    13.675867,
                    14.303982,
                    104.5632,
                    144.62547,
                    243.2081,
                    267.76,
                    267.76,
                    267.76,
                    267.76,
                ]
            ),
            "_xmax": np.array(
                [
                    283.46,
                    283.46,
                    283.46,
                    283.46,
                    283.46,
                    283.46,
                    283.46,
                    283.46,
                    283.46,
                    283.74622,
                    284.43222,
                    284.49908,
                    284.46643,
                    284.37195,
                ]
            ),
            "_time_ref": np.array([1]),
            "_n_f": 14,
            "_f": np.array(
                [
                    22.24,
                    23.04,
                    23.84,
                    25.44,
                    26.24,
                    27.84,
                    31.4,
                    51.26,
                    52.28,
                    53.86,
                    54.94,
                    56.66,
                    57.3,
                    58.0,
                ]
            ),
            "_n_ang": 10,
            "_ang": np.array([90.0, 30.0, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2]),
        }
        for key, value in data.items():
            assert isinstance(self.header[key], type(value))
            if isinstance(value, int):
                assert self.header[key] == value
            else:
                assert_array_almost_equal(self.header[key], np.array(value), decimal=2)

    def test_data(self):
        assert isinstance(self.data, dict)
        expected_data_keys = {"time", "rain", "temp_sfc", "tb"}
        assert set(self.data.keys()) == expected_data_keys
        assert len(self.data["time"]) == self.header["n"]
        assert self.data["time"][0] == 702432050
        assert self.data["time"][-1] == 702517849
        assert np.isclose(self.data["rain"], 4).all()
        assert np.isclose(np.mean(self.data["temp_sfc"]), 276.2, atol=1)
        assert np.isclose(np.mean(self.data["tb"]), 195.8, atol=1)
        assert self.data["tb"].shape == (
            self.header["n"],
            self.header["_n_f"],
            self.header["_n_ang"],
        )
        assert np.isclose(self.data["tb"][0, 0, 0], 28.307, atol=0.01)
        assert np.isclose(self.data["tb"][-1, 0, 0], 23.304, atol=0.01)
        assert np.isclose(self.data["tb"][0, -1, 0], 274.592, atol=0.01)
        assert np.isclose(self.data["tb"][0, 0, -1], 231.091, atol=0.01)


class TestIrtFileReading:
    header, data = rpg_bin.read_irt(f"{dir_name}/data/hyytiala/230406.IRT")

    def test_header(self):
        assert isinstance(self.header, dict)
        data = {
            "_code": np.array([671112000]),
            "n": 73317,
            "_xmin": np.array([-75.89093]),
            "_xmax": np.array([-35.11674]),
            "_time_ref": np.array([1]),
            "_n_f": 1,
            "_f": np.array([10.5]),
        }
        for key, value in data.items():
            assert isinstance(self.header[key], type(value))
            if isinstance(value, int):
                assert self.header[key] == value
            else:
                assert_array_almost_equal(self.header[key], np.array(value), decimal=2)

    def test_data(self):
        assert isinstance(self.data, dict)
        expected_data_keys = {
            "time",
            "rain",
            "irt",
            "_angles",
            "ir_elevation_angle",
            "ir_azimuth_angle",
        }
        assert set(self.data.keys()) == expected_data_keys
        assert len(self.data["time"]) == self.header["n"]
        assert self.data["time"][0] == 702432051
        assert self.data["time"][-1] == 702518389
        assert np.isclose(self.data["rain"], 0).all()
        assert np.isclose(self.data["ir_elevation_angle"], 89.95).all()
        assert np.isclose(self.data["ir_azimuth_angle"], 0.02).all()
        assert self.data["irt"].shape == (self.header["n"], self.header["_n_f"])
        assert np.isclose(np.mean(self.data["irt"]), 202.90, atol=1)


class TestHkdFileReading:
    header, data = rpg_bin.read_hkd(f"{dir_name}/data/hyytiala/230406.HKD")

    def test_header(self):
        assert isinstance(self.header, dict)
        data = {
            "_code": np.array([837854832]),
            "n": 78917,
            "_time_ref": np.array([1]),
            "_sel": np.array([831]),
        }
        for key, value in data.items():
            assert isinstance(self.header[key], type(value))
            if isinstance(value, int):
                assert self.header[key] == value
            else:
                assert_array_almost_equal(self.header[key], np.array(value), decimal=2)

    def test_data(self):
        assert isinstance(self.data, dict)
        expected_data_keys = {
            "time",
            "alarm",
            "latitude",
            "longitude",
            "temp",
            "stab",
            "flash",
            "qual",
            "status",
        }
        assert set(self.data.keys()) == expected_data_keys
        assert len(self.data["time"]) == self.header["n"]
        assert self.data["time"][0] == 702432002
        assert self.data["time"][-1] == 702518400
        assert np.isclose(self.data["alarm"], 0).all()
        assert np.isclose(self.data["qual"], 0).all()
        assert np.isclose(self.data["flash"], 15211, atol=1).all()
        assert np.isclose(self.data["latitude"], 61.84, atol=0.1).all()
        assert np.isclose(self.data["longitude"], 24.28, atol=0.1).all()
        assert self.data["temp"].shape == (self.header["n"], 4)
        assert self.data["stab"].shape == (self.header["n"], 2)
        assert np.isclose(np.mean(self.data["temp"]), 307.355, atol=1)
        assert np.sum(self.data["status"][:10]) == 968948470


class TestMetFileReading:
    header, data = rpg_bin.read_met(f"{dir_name}/data/hyytiala/230406.MET")

    def test_header(self):
        assert isinstance(self.header, dict)
        data = {
            "_code": np.array([599658944]),
            "n": 78917,
            "_air_pressure_min": 1010.5,
            "_air_pressure_max": 1012.4,
            "_air_temperature_min": 267.66,
            "_air_temperature_max": 283.56,
            "_relative_humidity_min": 41.5,
            "_relative_humidity_max": 87.6,
            "_wind_speed_min": 0.1,
            "_wind_speed_max": 21.7,
            "_wind_direction_min": 0.026275635,
            "_wind_direction_max": 358.02628,
            "_rainfall_rate_min": 0,
            "_rainfall_rate_max": 0,
            "_time_ref": np.array([1]),
        }
        for key, value in data.items():
            if isinstance(value, int):
                assert self.header[key] == value
            else:
                assert_array_almost_equal(self.header[key], np.array(value), decimal=2)

    def test_data(self):
        assert isinstance(self.data, dict)
        expected_data_keys = {
            "time",
            "rain",
            "air_pressure",
            "air_temperature",
            "relative_humidity",
            "wind_speed",
            "wind_direction",
            "rainfall_rate",
        }
        assert set(self.data.keys()) == expected_data_keys
        assert len(self.data["time"]) == self.header["n"]
        assert self.data["time"][0] == 702432002
        assert self.data["time"][-1] == 702518400
        assert np.isclose(self.data["rain"], 0).all()
        assert np.isclose(np.mean(self.data["air_temperature"]), 276.17, atol=0.1)
        assert np.isclose(np.mean(self.data["air_pressure"]), 1011.59, atol=0.1)
        assert np.isclose(np.mean(self.data["relative_humidity"]), 0.63, atol=0.1)
        assert (np.isclose(self.data["wind_speed"][0], 1.4, atol=0.1)).all()
        assert (np.isclose(self.data["wind_direction"][0], 129.03, atol=0.1)).all()
        assert (np.isclose(self.data["rainfall_rate"][0], 0, atol=0.1)).all()
        assert (np.isclose(self.data["wind_speed"][-1], 1.3, atol=0.1)).all()
        assert (np.isclose(self.data["wind_direction"][-1], 207.03, atol=0.1)).all()
        assert (np.isclose(self.data["rainfall_rate"][-1], 0, atol=0.1)).all()


@pytest.mark.parametrize(
    "a,b",
    [
        (-12245.50, -122 - 45 / 60 - 30 / 60 / 60),
        (-3321.25, -33 - 21 / 60 - 15 / 60 / 60),
        (12245.50, 122 + 45 / 60 + 30 / 60 / 60),
        (3321.25, 33 + 21 / 60 + 15 / 60 / 60),
    ],
)
def test_decode_latlon(a, b):
    assert np.isclose(rpg_bin._decode_latlon(a), b)
