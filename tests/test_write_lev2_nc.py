import os

from mwrpy import lev1_to_nc
from mwrpy.level2.write_lev2_nc import lev2_to_nc

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{PACKAGE_DIR}/data"
COEFFICIENTS_DIR = f"{PACKAGE_DIR}/../mwrpy/site_config/"
DATE = "2023-04-06"


def test_level2_processing():
    lev1_file = "temp_file1.nc"
    lev2_file = "temp_file2.nc"
    temp_file = "temp_file3.nc"
    hum_file = "temp_file4.nc"

    site = "hyytiala"

    lev1_to_nc(site, "1C01", DATA_DIR, lev1_file)

    # mwr-single
    lev2_to_nc(site, "2I01", lev1_file, lev2_file)
    lev2_to_nc(site, "2I02", lev1_file, lev2_file)
    lev2_to_nc(site, "2P01", lev1_file, lev2_file)
    lev2_to_nc(site, "2P03", lev1_file, hum_file)

    # mwr-multi
    lev2_to_nc(site, "2P02", lev1_file, temp_file)
    lev2_to_nc(
        site, "2P04", lev1_file, lev2_file, temp_file=temp_file, hum_file=hum_file
    )
    lev2_to_nc(
        site, "2P07", lev1_file, lev2_file, temp_file=temp_file, hum_file=hum_file
    )
    lev2_to_nc(
        site, "2P08", lev1_file, lev2_file, temp_file=temp_file, hum_file=hum_file
    )

    for file in (lev1_file, lev2_file, temp_file, hum_file):
        if os.path.exists(file):
            try:
                os.remove(file)
            except PermissionError:
                pass
