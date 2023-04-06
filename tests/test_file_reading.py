import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from mwrpy.level1 import rpg_bin

dir_name = os.path.dirname(os.path.realpath(__file__))


class TestBrtFileReading:
    header, data = rpg_bin.read_brt(f"{dir_name}/data/230406.BRT")

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
            "n": 21389,
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
            assert type(self.header[key]) == type(value)
            if isinstance(value, int):
                assert self.header[key] == value
            else:
                assert_array_almost_equal(self.header[key], value, decimal=2)

    def test_data(self):
        assert isinstance(self.data, dict)
        expected_data_keys = {"time", "rain", "tb", "elevation_angle", "azimuth_angle"}
        assert set(self.data.keys()) == expected_data_keys
        assert len(self.data["time"]) == self.header["n"]
        assert self.data["time"][0] == 702432051
        assert self.data["time"][-1] == 702457199
        assert np.isclose(self.data["azimuth_angle"], 0.02).all()
        assert np.isclose(self.data["elevation_angle"], 90.01).all()
        assert np.isclose(self.data["rain"], 0).all()
        assert self.data["tb"].shape == (self.header["n"], self.header["_n_f"])
        assert np.min(self.data["tb"]) > 15
        assert np.max(self.data["tb"]) < 280
        assert np.isclose(np.mean(self.data["tb"]), 124.37, atol=1)
