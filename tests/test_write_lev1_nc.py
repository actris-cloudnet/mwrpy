import datetime
import os

from mwrpy import lev1_to_nc

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{PACKAGE_DIR}/data"
COEFFICIENTS_DIR = f"{PACKAGE_DIR}/../mwrpy/site_config/"
DATE = "2023-04-06"


def test_lev1_to_nc():
    for site in next(os.walk(COEFFICIENTS_DIR))[1]:
        hatpro = lev1_to_nc(site, "1C01", DATA_DIR)
        assert hatpro.date == DATE
        for t in hatpro.data["time"][:]:
            date = str(datetime.datetime.utcfromtimestamp(t).date())
            assert date == DATE
