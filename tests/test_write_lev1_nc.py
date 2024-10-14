import datetime
import os

import netCDF4
import numpy as np

from mwrpy.level1.write_lev1_nc import lev1_to_nc

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{PACKAGE_DIR}/data/hyytiala"
COEFFICIENTS_DIR = f"{PACKAGE_DIR}/../mwrpy/site_config/"
DATE = "2023-04-06"
site = "hyytiala"
product_list = ["1B01", "1B11", "1B21", "1C01"]


def test_lev1_to_nc():
    for prod in product_list:
        hatpro = lev1_to_nc(prod, DATA_DIR, site)
        assert hatpro.date == DATE
        for t in hatpro.data["time"][:]:
            date = str(datetime.datetime.utcfromtimestamp(t).date())
            assert date == DATE


def test_output_nc_file():
    for prod in product_list:
        temp_file = "temp_file.nc"
        lev1_to_nc(prod, DATA_DIR, site, output_file=temp_file)
        with netCDF4.Dataset(temp_file) as nc:
            # Write tests for the created netCDF file here:
            assert nc.date == DATE
            if prod in ["1B01", "1C01"]:
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
            elif prod == "1B11":
                assert 1.05e-05 in nc.variables["ir_wavelength"][:].data
            else:
                var_names = ["air_pressure", "air_temperature", "relative_humidity"]
                assert len(set(var_names) & set(nc.variables)) == 3
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except PermissionError:
                pass
