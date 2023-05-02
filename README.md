# MWRpy

[![Tests](https://github.com/actris-cloudnet/mwrpy/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/mwrpy/actions/workflows/test.yml)

### Usage

```python
import mwrpy
hatpro = mwrpy.lev1_to_nc("hyytiala", "1C01", "/path/to/files/")
```


### Licence

MIT


[//]: # (MWRpy is a Python based software to process RPG Microwave Radiometer data and is developed at the University of Cologne, Germany as part of the [Aerosol, Clouds and Trace Gases Research Infrastructure &#40;ACTRIS&#41;]&#40;https://actris.eu/&#41;.)

[//]: # (The software features reading raw data, Level 1 quality control, generation of Level 2 data products and visualization.)

[//]: # ()
[//]: # (The data format including metadata information, variable names and file naming is designed to be compliant with the data structure and naming convention developed in the [EUMETNET Profiling Programme E-PROFILE]&#40;https://www.eumetnet.eu/&#41;.)

[//]: # ()
[//]: # (![MWRpy example output]&#40;https://atmos.meteo.uni-koeln.de/~hatpro/quicklooks/obs/site/jue/tophat/actris/level2/2022/10/29/20221029_juelich_temperature.png&#41;)

[//]: # ()
[//]: # (## MWRpy Structure)

[//]: # ()
[//]: # (`mwrpy/rpg_mwr.py` contains the base class <b>RpgArray</b> for storing variables as netCDF4.)

[//]: # ()
[//]: # (### `mwrpy/site_config/`)

[//]: # ()
[//]: # (This folder contains configuration files for each instrument type in `instrument.yaml` and subfolders for each site, where retrieval coeffiecients are stored in `coefficients/` and `config.yaml` defines site specific information &#40;including input and output data paths&#41;, which needs to be modified and is used for processing purposes and metadata generation.)

[//]: # ()
[//]: # (### `mwrpy/level1/`)

[//]: # ()
[//]: # (<b>*lev1_to_nc*</b> in `write_lev1.py` reads the raw binary files &#40;.BRT, .BLB/.BLS, .IRT, .MET, .HKD&#41; stored in the same folder containing data of one day, applies quality control &#40;`quality_control.py`&#41; and writes it into a netCDF4 file using metadata defined in `lev1_meta_nc.py`.)

[//]: # ()
[//]: # (#### Quality flags &#40;bit variable&#41;)

[//]: # (    # Bit 1: missing_tb)

[//]: # (    # Bit 2: tb_below_threshold)

[//]: # (    # Bit 3: tb_above_threshold)

[//]: # (    # Bit 4: spectral_consistency_above_threshold)

[//]: # (    # Bit 5: receiver_sanity_failed)

[//]: # (    # Bit 6: rain_detected)

[//]: # (    # Bit 7: sun_in_beam)

[//]: # (    # Bit 8: tb_offset_above_threshold)

[//]: # ()
[//]: # (#### Level 1 Data Types)

[//]: # (* 1B01: MWR brightnesss temperatures from .BRT and .BLB/.BLS files)

[//]: # (* 1B11: IR brightnesss temperatures from .IRT files)

[//]: # (* 1B21: Weather station data from .MET files)

[//]: # (* 1C01: Combined data type with time corresponding to 1B01)

[//]: # ()
[//]: # (### `mwrpy/level2/`)

[//]: # ()
[//]: # (<b>*lev2_to_nc*</b> in `write_lev2.py` reads Level 1 files, applies retrieval coefficients read in by `get_ret_coeff.py` for Level 2 products and writes it into a netCDF4 file using metadata defined in `lev2_meta_nc.py`. For the LWP product an offset correction is applied &#40;`lwp_offset.py`&#41;.)

[//]: # ()
[//]: # (#### Level 2 Data Types)

[//]: # (* 2I01: Liquid water path &#40;LWP&#41;)

[//]: # (* 2I02: Integrated water vapor &#40;IWV&#41;)

[//]: # (* 2P01: Temperature profiles from single-pointing observations)

[//]: # (* 2P02: Temperature profiles from multiple-pointing observations)

[//]: # (* 2P03: Absolute humidity profiles)

[//]: # (* 2P04: Relative humidity profiles &#40;derived from 2P01/2P02 + 2P03&#41;)

[//]: # (* 2P07: Potential temperature &#40;derived from 2P01/2P02 + 2P03&#41;)

[//]: # (* 2P08: Equivalent potential temperature &#40;derived from 2P01/2P02 + 2P03&#41;)

[//]: # (* 2S02: Brightness temperature spectrum)

[//]: # ()
[//]: # (### `mwrpy/plots/`)

[//]: # ()
[//]: # (<b>*generate_figure*</b> in `generate_plots.py` creates .png figures using plot specific metadata defined in `plot_meta.py`.)

[//]: # ()
[//]: # (<b>*generate_stat*</b> in `stat_plot.py` creates statistic plots &#40;called by `stats` as product, see below&#41; as .png figures using plot specific metadata defined in `stat_meta.py`.)

[//]: # ()
[//]: # (## How to run the software)

[//]: # ()
[//]: # (Running the software is based on a wrapper script `mwrpy/mwrpy.py`:)

[//]: # ()
[//]: # (    usage: mwrpy.py [-h] -s SITE [-d YYYY-MM-DD] [--start YYYY-MM-DD])

[//]: # (                           [--stop YYYY-MM-DD] [-p ...] COMMAND ...)

[//]: # ()
[//]: # ()
[//]: # (The following general arguments can be used and must be issued before the command argument:)

[//]: # ()
[//]: # ()
[//]: # (| Short | Long         | Default           | Description                                                                        |)

[//]: # (|:------|:-------------|:------------------|:-----------------------------------------------------------------------------------|)

[//]: # (| `-h`  | `--help`     |                   | Show help and exit.                                                                |)

[//]: # (| `-s`  | `--site`     |                   | Site to process data from, e.g, `hyytiala`. Required.                              |)

[//]: # (| `-d`  | `--date`     |                   | Single date to be processed. Alternatively, `--start` and `--stop` can be defined. |)

[//]: # (|       | `--start`    | `current day - 1` | Starting date.                                                                     |)

[//]: # (|       | `--stop`     | `current day `    | Stopping date.                                                                     |)

[//]: # (| `-p`  | `--products` | all               | Processed products, e.g, `1C01, 2I02, 2P03, stats`.                                |)

[//]: # ()
[//]: # ()
[//]: # (### Commands)

[//]: # ()
[//]: # (### `process`)

[//]: # ()
[//]: # (The `process` command processes standard MWR products using the script `mwrpy/process_mwrpy.py`, where functions for generating and visualizing Level 1 and Level 2 products are called &#40;<b>*lev1_to_nc*, *lev2_to_nc*, *generate_figure*</b>&#41;.)

[//]: # ()
[//]: # (In addition to the general arguments, it accepts the following special arguments.)

[//]: # ()
[//]: # (| Short | Long                   | Default | Description                                                                        |)

[//]: # (|:------|:-----------------------|:--------|:-----------------------------------------------------------------------------------|)

[//]: # (| `-f`  | `--figure`             | `False` | Produce figures only; no processing.)
