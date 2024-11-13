# MWRpy

[![MWRpy tests](https://github.com/actris-cloudnet/mwrpy/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/mwrpy/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/mwrpy.svg)](https://badge.fury.io/py/mwrpy)
[![DOI](https://zenodo.org/badge/619528669.svg)](https://zenodo.org/doi/10.5281/zenodo.11614185)
[![paper](https://joss.theoj.org/papers/10.21105/joss.06733/status.svg)](https://doi.org/10.21105/joss.06733)

MWRpy is a Python software to process RPG Microwave Radiometer data and is developed
at the University of Cologne, Germany as part
of the [Aerosol, Clouds and Trace Gases Research Infrastructure (ACTRIS)](https://actris.eu/).

The software features reading raw data, Level 1 quality control, generation of
Level 2 data products and visualization and is based on the IDL code
[mwr_pro](https://zenodo.org/records/7973553).

The netCDF data format including metadata information, variable names and file naming
is designed to be compliant with the data structure and naming convention
developed in the [EUMETNET Profiling Programme E-PROFILE](https://www.eumetnet.eu/).

MWRpy documentation: <https://actris-cloudnet.github.io/mwrpy/>

![MWRpy example output](https://atmos.meteo.uni-koeln.de/~hatpro/quicklooks/obs/site/jue/tophat/actris/level2/2022/10/29/20221029_juelich_temperature.png)

## Citing

If you wish to acknowledge MWRpy in your publication, please cite:

> Marke et al., (2024). MWRpy: A Python package for processing microwave radiometer data. Journal of Open Source
> Software, 9(98), 6733, https://doi.org/10.21105/joss.06733

## Installation

From PyPI:

```shell
python3 -m pip install mwrpy
```

From GitHub:

```shell
git clone https://github.com/actris-cloudnet/mwrpy.git
cd mwrpy
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install .
```

MWRpy requires Python 3.10 or newer.

## Configuration

The folder `mwrpy/site_config/` contains configuration files for each instrument
type, which defines the input and output data paths etc.
For example, this is the [configuration file for RPG-HATPRO](mwrpy/site_config/hatpro.yaml).

The folders for each site, e.g. `mwrpy/site_config/hyytiala/`, contain a
site and instrument specific configuration file (`config.yaml`) and retrieval coefficients.
For example, this is the [configuration file for Hyytiälä](mwrpy/site_config/hyytiala/config.yaml).

## Command line usage

MWRpy can be run using the command line tool `mwrpy/cli.py`:

    usage: mwrpy/cli.py [-h] -s SITE [-d YYYY-MM-DD] [--start YYYY-MM-DD]
                           [--stop YYYY-MM-DD] [-p ...] [{process,plot}]

Arguments:

| Short | Long         | Default             | Description                                                                        |
| :---- | :----------- | :------------------ | :--------------------------------------------------------------------------------- |
| `-h`  | `--help`     |                     | Show help and exit.                                                                |
| `-s`  | `--site`     |                     | Site to process data from, e.g, `hyytiala`. Required.                              |
| `-d`  | `--date`     |                     | Single date to be processed. Alternatively, `--start` and `--stop` can be defined. |
|       | `--start`    | `current day - 1`   | Starting date.                                                                     |
|       | `--stop`     | `current day `      | Stopping date.                                                                     |
| `-p`  | `--products` | 1C01, single, multi | Processed products, e.g, `1C01, 2I02, 2P03, single`, see below.                    |

Commands:

| Command     | Description                                                |
| :---------- | :--------------------------------------------------------- |
| `process`   | Process data and generate plots (default).                 |
| `plot`      | Only generate plots.                                       |
| `no-plot`   | Only generate products.                                    |
| `reprocess` | Like `process`, but skips days when data processing fails. |

### Data types

#### Level 1

- 1B01: MWR brightness temperatures from .BRT and .BLB/.BLS files + retrieved spectrum
- 1B11: IR brightness temperatures from .IRT files
- 1B21: Weather station data from .MET files
- 1C01: Combined data type with time corresponding to 1B01

#### Level 2

- 2I01: Liquid water path (LWP)
- 2I02: Integrated water vapor (IWV)
- 2I06: Stability Indices
- 2P01: Temperature profiles from single-pointing observations
- 2P02: Temperature profiles from multiple-pointing observations
- 2P03: Absolute humidity profiles
- 2P04: Relative humidity profiles (derived from 2P01/2P02 + 2P03)
- 2P07: Potential temperature (derived from 2P01/2P02 + 2P03)
- 2P08: Equivalent potential temperature (derived from 2P01/2P02 + 2P03)
- single: Single pointing data product (including 2I01, 2I02, 2I06, 2P01, 2P03, and derived products)
- multi: Multiple pointing data product (including 2P02, and derived products)

## Licence

MIT
