from setuptools import find_packages, setup

version: dict = {}
with open("mwrpy/version.py", encoding="utf8") as f:
    exec(f.read(), version)  # pylint: disable=W0122

with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup(
    name="mwrpy",
    version=version["__version__"],
    description="Python package for Microwave Radiometer processing in ACTRIS",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="University of Cologne",
    author_email="actris-ccres-mwr@uni-koeln.de",
    url="https://github.com/actris-cloudnet/mwrpy",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "scipy",
        "netCDF4",
        "matplotlib",
        "metpy",
        "ephem",
        "pandas",
        "pyyaml",
        "timezonefinder",
    ],
    extras_require={
        "test": ["pylint", "mypy", "pytest", "pytest-flakefinder"],
    },
)
