import glob
import os
import tempfile

import pytest

from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.level2.lev2_collocated import generate_lev2_multi, generate_lev2_single

SITE = "hyytiala"
DATA_FORMAT = "e-profile"

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{PACKAGE_DIR}/data/{SITE}"
COEFFICIENTS_DIR = f"{PACKAGE_DIR}/../mwrpy/site_config/"
COEFF_FILES = glob.glob(f"{COEFFICIENTS_DIR}/{SITE}/coefficients/*.ret")


@pytest.fixture(scope="module")
def l1_file(request):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    lev1_to_nc("1C01", DATA_DIR, SITE, path)

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)
    return path


def test_generate_lev2_single_site(l1_file):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    generate_lev2_single(SITE, DATA_FORMAT, l1_file, path)
    os.unlink(path)


def test_generate_lev2_single_no_site(l1_file):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    generate_lev2_single(None, DATA_FORMAT, l1_file, path, coeff_files=COEFF_FILES)
    os.unlink(path)


def test_generate_lev2_multi_site(l1_file):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    generate_lev2_multi(SITE, DATA_FORMAT, l1_file, path)
    os.unlink(path)


def test_generate_lev2_multi_no_site(l1_file):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    generate_lev2_multi(None, DATA_FORMAT, l1_file, path, coeff_files=COEFF_FILES)
