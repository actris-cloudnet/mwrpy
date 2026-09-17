import datetime
import glob
import os
import sys

import netCDF4
import pytest

from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.level2.lev2_collocated import generate_lev2_multi, generate_lev2_single

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{PACKAGE_DIR}/data/hyytiala"
COEFF_FILES = glob.glob(
    f"{PACKAGE_DIR}/../mwrpy/site_config/hyytiala/coefficients/*.ret"
)
DATE = datetime.date(2023, 4, 6)
# Site without config file in the repository
INSTRUMENT_CONFIG = {
    "site": "nowhere",
    "latitude": 61.844,
    "longitude": 24.287,
    "altitude": 150,
}


@pytest.fixture(scope="module")
def l1_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("no_config") / "l1c.nc"
    hatpro = lev1_to_nc(
        "1C01",
        DATA_DIR,
        output_file=path,
        coeff_files=COEFF_FILES,
        instrument_config=INSTRUMENT_CONFIG,
        instrument_type="hatpro",
    )
    # Returned object keeps epoch seconds even though file has hours
    t0 = float(hatpro.data["time"][:][0])
    assert datetime.datetime.fromtimestamp(t0, tz=datetime.timezone.utc).date() == DATE
    assert "time_bnds" in hatpro.data
    return path


def test_lev1(l1_file):
    with netCDF4.Dataset(l1_file) as nc:
        assert nc.location == "nowhere"
        assert nc.cloudnet_file_type == "mwr-l1c"
        assert (nc.year, nc.month, nc.day) == ("2023", "04", "06")
        assert nc.variables["time"].units.startswith("hours since 2023-04-06")
        assert 0 <= nc.variables["time"][:].min() < nc.variables["time"][:].max() <= 24
        assert "time_bnds" not in nc.variables
        assert "zenith_angle" in nc.variables
        assert nc.variables["altitude"][:].max() == 150


# Level 2 writing is not tested on Windows
@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on Windows")
@pytest.mark.parametrize(
    "fun, file_type, variable",
    [
        (generate_lev2_single, "mwr-single", "lwp"),
        (generate_lev2_multi, "mwr-multi", "temperature"),
    ],
)
def test_lev2(l1_file, tmp_path, fun, file_type, variable):
    path = tmp_path / "l2.nc"
    fun(l1_file, path, coeff_files=COEFF_FILES)
    with netCDF4.Dataset(path) as nc:
        assert nc.location == "nowhere"
        assert nc.cloudnet_file_type == file_type
        assert variable in nc.variables
        assert nc.variables["time"].units.startswith("hours since 2023-04-06")
