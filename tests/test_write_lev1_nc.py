import datetime
import os

import netCDF4
import numpy as np

from mwrpy import lev1_to_nc

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{PACKAGE_DIR}/data/hyytiala"
COEFFICIENTS_DIR = f"{PACKAGE_DIR}/../mwrpy/site_config/"
DATE = "2023-04-06"


def test_lev1_to_nc():
    for site in next(os.walk(COEFFICIENTS_DIR))[1]:
        hatpro = lev1_to_nc(site, "1C01", DATA_DIR)
        assert hatpro.date == DATE
        for t in hatpro.data["time"][:]:
            date = str(datetime.datetime.utcfromtimestamp(t).date())
            assert date == DATE


def test_output_nc_file():
    temp_file = "temp_file.nc"
    lev1_to_nc("hyytiala", "1C01", DATA_DIR, temp_file)
    with netCDF4.Dataset(temp_file) as nc:
        # Write tests for the created netCDF file here:
        assert nc.date == DATE
        assert np.allclose(
            nc.variables["frequency"][:],
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
                58,
            ],
        )
    os.remove(temp_file)
