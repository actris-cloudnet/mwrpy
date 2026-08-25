"""Module for processing."""

import datetime
import glob
import logging
import os
import time
from typing import Literal

import netCDF4 as nc
import pandas as pd

import mwrpy.utils
from mwrpy.level1.write_lev1_nc import lev1_to_nc, prepare_data
from mwrpy.level2.lev2_collocated import (
    generate_lev2_lhumpro,
    generate_lev2_multi,
    generate_lev2_single,
)
from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.plots.generate_plots import generate_figure

PRODUCT_NAME = {
    "1B01": [
        "tb",
        "tb_spectrum",
        "sen",
        "quality_flag",
        "hkd",
    ],
    "1B11": [
        "irt",
    ],
    "1B21": [
        "met",
        "met2",
        "met_quality_flag",
    ],
    "1C01": [
        "tb",
        "tb_spectrum",
        "sen",
        "irt",
        "met",
        "met2",
        "quality_flag",
        "met_quality_flag",
        "hkd",
        "cov",
        "his",
    ],
    "2I01": ["lwp", "lwp_scan"],
    "2I02": ["iwv", "iwv_scan"],
    "2I06": ["stability"],
    "2P01": ["temperature"],
    "2P02": ["temperature"],
    "2P03": ["absolute_humidity"],
    "2P04": ["relative_humidity"],
    "2P07": ["potential_temperature"],
    "2P08": "equivalent_potential_temperature",
    "single": [
        "lwp",
        "lwp_scan",
        "iwv",
        "iwv_scan",
        "stability",
        "temperature",
        "absolute_humidity",
        "relative_humidity",
        "potential_temperature",
        "equivalent_potential_temperature",
    ],
    "multi": [
        "temperature",
        "relative_humidity",
        "potential_temperature",
        "equivalent_potential_temperature",
    ],
}
f_names_stability = list(
    [
        "cape",
        "k_index",
        "total_totals",
        "lifted_index",
        "showalter_index",
        "ko_index",
    ]
)
IType = Literal["hatpro", "lhatpro", "lhumpro_u90"]


def main(args):
    """Main function for processing and plotting MWR data."""
    logging.basicConfig(level="INFO")
    _start_date, _stop_date = mwrpy.utils.get_processing_dates(args)
    start_date = mwrpy.utils.isodate2date(_start_date)
    stop_date = mwrpy.utils.isodate2date(_stop_date)

    for date in mwrpy.utils.date_range(start_date, stop_date):
        for product in args.products:
            if product not in PRODUCT_NAME:
                logging.error(f"Product {product} not recognised")
                continue
            if args.format == "cloudnet":
                if product not in ("1C01", "single", "multi"):
                    logging.error(
                        f"Product {product} not available in cloudnet format. Skipping."
                    )
                    continue
                if args.altitude is None:
                    logging.info("Site altitude not provided. Taking default of 0 m.")
            start = time.process_time()
            if args.command != "plot":
                logging.info(f"Processing {product} product, {args.site} {date}")
                if args.command == "reprocess":
                    try:
                        output_file = process_product(
                            product,
                            date,
                            args.site,
                            args.format,
                            args.instrument,
                            args.altitude,
                            args.azimuth_offset,
                        )
                    except Exception as e:
                        logging.error(
                            f"Error in processing products: {e}. Incomplete or no processing for {date}."
                        )
                        output_file = None
                else:
                    output_file = process_product(
                        product,
                        date,
                        args.site,
                        args.format,
                        args.instrument,
                        args.altitude,
                        args.azimuth_offset,
                    )
                if output_file:
                    logging.info("Processed %s: %s", product, output_file)

            if args.command != "no-plot":
                logging.info(f"Plotting {product} product, {args.site} {date}")
                plot_product(product, date, args.site, args.format, args.instrument)

            elapsed_time = time.process_time() - start
            logging.info(f"Processing took {elapsed_time:.1f} seconds")


def process_product(
    prod: str,
    date: datetime.date,
    site: str,
    data_format: str,
    instrument: IType,
    altitude: float,
    azimuth_offset: float | None,
):
    """Process a given product for a specific date and site.
    This function handles the processing of different products based on their type
    (level 1, level 2, single, multi) and manages the necessary file
    operations and directory structures.

    Args:
        prod: Product code (e.g., '1C01', '2I01', 'single', 'multi').
        date: Date for which the product is to be processed.
        site: Site identifier.
        data_format: Data format of the netCDF file (cloudnet, e-profile).
        instrument: Specific instrument type (hatpro, lhatpro, etc.).
        altitude: Altitude of the site in meters above mean sea level.
        azimuth_offset: Azimuth offset to be added to azimuth angle.

    Returns:
        output_file: Name of output file.
    """
    filename = getattr(
        mwrpy.utils,
        "_get_filename_cloudnet" if data_format == "cloudnet" else "_get_filename",
    )
    output_file = filename(prod, date, site, instrument)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Check for LWP offset from previous/next days
    lwp_offset: list[float | None] = [None, None]
    for iday in range(3):
        xday = [
            date - datetime.timedelta(days=iday + 1),
            date + datetime.timedelta(days=iday + 1),
        ]
        offset_file = [
            filename("lwp_offset", xday[0], site, instrument),
            filename("lwp_offset", xday[1], site, instrument),
        ]
        if (
            (prod in ("2I01", "single"))
            and (os.path.isfile(offset_file[0]))
            and (lwp_offset[0] is None)
        ):
            csv_off = pd.read_csv(offset_file[0], usecols=["date", "offset"])
            if xday[0].strftime("%m-%d") in csv_off["date"].values:
                lwp_offset[0] = csv_off.loc[
                    csv_off["date"] == xday[0].strftime("%m-%d"), "offset"
                ].values[0]
        if (
            (prod in ("2I01", "single"))
            and (os.path.isfile(offset_file[1]))
            and (lwp_offset[1] is None)
        ):
            csv_off = pd.read_csv(offset_file[1], usecols=["date", "offset"])
            if xday[1].strftime("%m-%d") in csv_off["date"].values:
                lwp_offset[1] = csv_off.loc[
                    csv_off["date"] == xday[1].strftime("%m-%d"), "offset"
                ].values[0]
    lwp_offset_tuple = (lwp_offset[0], lwp_offset[1])

    l1_filename = filename("1C01", date, site, instrument)
    # Process level 1 data
    if prod[0] == "1":
        params = mwrpy.utils.read_config(site, instrument, "params")
        if data_format == "e-profile":
            altitude = params["altitude"]
            if altitude is None:
                altitude = 0.0
                logging.info("Site altitude not provided. Taking default of 0 m.")
            azimuth_offset = (
                params["azimuth_offset"]
                if "azimuth_offset" in params and params["azimuth_offset"] is not None
                else 0.0
            )
        lev1_to_nc(
            prod,
            _get_raw_file_path(date, site, instrument)
            if instrument is None
            else _get_raw_file_path(date, None, instrument),
            data_format,
            site=site,
            output_file=output_file,
            lidar_path=_get_lidar_file_path(date, site, params),
            date=date,
            instrument_type=instrument,
            altitude=altitude,
            azimuth_offset=azimuth_offset,
        )

    # Process level 2 single products
    elif prod[0] == "2":
        if prod in ("2P04", "2P07", "2P08"):
            temp_file = filename("2P02", date, site, instrument)
            if len(temp_file) == 0:
                temp_file = filename("2P01", date, site, instrument)
            hum_file = filename("2P03", date, site, instrument)
        else:
            temp_file = None
            hum_file = None
        lev2_to_nc(
            prod,
            filename("1C01", date, site, instrument),
            data_format,
            output_file=output_file,
            site=site,
            temp_file=temp_file,
            hum_file=hum_file,
            lwp_offset=lwp_offset_tuple,
            instrument_type=instrument,
        )

    # Process level 2 combined products
    elif prod == "single" and instrument != "lhumpro_u90":
        generate_lev2_single(
            site,
            data_format,
            l1_filename,
            output_file,
            lwp_offset_tuple,
            None,
            instrument,
        )
    elif instrument == "lhumpro_u90":
        generate_lev2_lhumpro(
            site,
            data_format,
            l1_filename,
            output_file,
            lwp_offset_tuple,
            None,
            instrument,
        )
    elif prod == "multi":
        generate_lev2_multi(
            site,
            data_format,
            l1_filename,
            output_file,
            None,
            instrument,
        )

    # Update LWP offset file if necessary
    offset_current = filename("lwp_offset", date, site, instrument)
    if (
        (prod in ("2I01", "single"))
        and (os.path.isfile(output_file))
        and (os.path.isfile(offset_current))
    ):
        output = nc.Dataset(output_file)
        if (
            (round(float(output["lwp_offset"][:].mean()), 5) not in lwp_offset)
            and (round(float(output["lwp_offset"][:].mean()), 5) != 0.0)
            and (abs(round(float(output["lwp_offset"][:].mean()), 5)) < 0.1)
        ):
            csv_off = pd.read_csv(offset_current, usecols=["date", "offset"])
            csv_off = pd.concat(
                [
                    csv_off,
                    pd.DataFrame(
                        {
                            "date": date.strftime("%m-%d"),
                            "offset": round(float(output["lwp_offset"][:].mean()), 5),
                        },
                        index=[0],
                    ),
                ]
            )
            csv_off = csv_off.sort_values(by=["date"])
            csv_off = csv_off.drop_duplicates(subset=["date"])
            csv_off.to_csv(offset_current, index=False)
    elif (
        (prod in ("2I01", "single"))
        and (os.path.isfile(output_file))
        and (not os.path.isfile(offset_current))
    ):
        output = nc.Dataset(output_file)
        if (round(float(output["lwp_offset"][:].mean()), 5) != 0.0) and (
            abs(round(float(output["lwp_offset"][:].mean()), 5)) < 0.1
        ):
            csv_off = pd.DataFrame(
                {
                    "date": date.strftime("%m-%d"),
                    "offset": round(float(output["lwp_offset"][:].mean()), 5),
                },
                index=[0],
            )
            csv_off.to_csv(offset_current, index=False)

    return output_file


def plot_product(prod: str, date, site: str, data_format: str, instrument: IType):
    """Plot a given product for a specific date and site.
    Plotting covariance data without 1C01 file is supported.

    Args:
        prod: Product code (e.g., '1C01', '2I01', 'single', 'multi').
        date: Date for which the product is to be plotted.
        site: Site identifier.
        data_format: Data format of the netCDF file (cloudnet, e-profile).
        instrument: Specific instrument type (hatpro, lhatpro, etc.).

    Returns:
        None
    """
    filename = getattr(
        mwrpy.utils,
        "_get_filename_cloudnet" if data_format == "cloudnet" else "_get_filename",
    )
    input_file = filename(prod, date, site, instrument)
    if not os.path.isfile(input_file):
        logging.warning("Nothing to plot for product " + prod)
    params = mwrpy.utils.read_config(site, instrument, "params")
    output_dir = f"{os.path.dirname(input_file)}/"

    # Plot level 1 data
    if os.path.isfile(input_file) and prod[0] == "1":
        keymap = {
            "tb": ["tb"],
            "tb_spectrum": ["tb_spectrum"],
            "sen": ["elevation_angle", "azimuth_angle"],
            "irt": ["irt"],
            "met": ["air_temperature", "relative_humidity", "rainfall_rate"],
            "met2": ["air_pressure", "wind_direction", "wind_speed"],
            "quality_flag": ["quality_flag"],
            "met_quality_flag": ["met_quality_flag"],
            "hkd": ["t_amb", "t_rec", "t_sta"],
            "cov": ["tb_cov_ln2", "tb_cov_amb"],
            "his": ["tb_cov_ln2", "tb_cov_amb", "Gain"],
        }
        for key in PRODUCT_NAME[prod]:
            variables = keymap[key]
            ele_range = (
                (
                    89.0,
                    91.0,
                )
                if key in ("tb", "tb_spectrum", "irt")
                else (-1.0, 91.0)
            )
            if key == "his":
                his_data = prepare_data(
                    "", key, params, None, date=time.mktime(date.timetuple())
                )
                assert isinstance(his_data, dict)
                if len(his_data) > 0:
                    generate_figure(
                        "",
                        ["tb_cov_ln2", "tb_cov_amb", "Gain"]
                        if "tb_cov_ln2" in his_data
                        else ["Gain"],
                        save_path=params["path_to_cal"],
                        image_name=key,
                        instrument_type=instrument,
                        cov_data=his_data,
                        site=site,
                    )
                else:
                    logging.warning("Nothing to plot for product " + prod)
            else:
                output_dir = params["path_to_cal"] if key == "cov" else output_dir
                if output_dir is not None:
                    generate_figure(
                        input_file,
                        variables,
                        ele_range=ele_range,
                        save_path=output_dir + "COVARIANCE/"
                        if key == "cov"
                        else output_dir,
                        image_name=key,
                        instrument_type=instrument,
                    )

    # Plot level 2 single products
    elif os.path.isfile(input_file) and (prod[0] == "2"):
        for key in PRODUCT_NAME[prod]:
            elevation = (
                (
                    89.0,
                    91.0,
                )
                if prod in ("2P01", "2P03", "2I06") or key in ("lwp", "iwv")
                else (
                    -1.0,
                    91.0,
                )
            )
            pointing = 1 if prod in ("2P02", "2P04", "2P07", "2P08") else 0
            if prod == "2I06":
                f_names = f_names_stability
                generate_figure(
                    input_file,
                    f_names,
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=PRODUCT_NAME[prod][0],
                    title=False,
                    instrument_type=instrument,
                )
            elif key in ("lwp_scan", "iwv_scan"):
                generate_figure(
                    input_file,
                    [key.rstrip("_scan")],
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=key,
                    title=False,
                    instrument_type=instrument,
                )
            else:
                generate_figure(
                    input_file,
                    [key],
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=key,
                    pointing=pointing,
                    instrument_type=instrument,
                )

    # Plot level 2 combined products
    elif os.path.isfile(input_file) and (prod in ("single", "multi")):
        for var_name in PRODUCT_NAME[prod]:
            elevation = (
                (
                    -1.0,
                    91.0,
                )
                if prod == "multi" or var_name in ("lwp_scan", "iwv_scan")
                else (
                    89.0,
                    91.0,
                )
            )
            pointing = 1 if prod == "multi" else 0
            f_names = f_names_stability
            if var_name == "stability":
                keymap = {
                    var_name: f_names_stability,
                }
            else:
                keymap = {
                    var_name: [var_name],
                }
            title = (
                False
                if var_name in f_names or var_name in ("lwp_scan", "iwv_scan")
                else True
            )
            for key, variables in keymap.items():
                if key in ("lwp_scan", "iwv_scan"):
                    generate_figure(
                        input_file,
                        [key.rstrip("_scan")],
                        ele_range=elevation,
                        save_path=output_dir,
                        image_name=key,
                        title=False,
                        instrument_type=instrument,
                    )
                else:
                    generate_figure(
                        input_file,
                        variables,
                        ele_range=elevation,
                        save_path=output_dir,
                        image_name=key,
                        title=title,
                        pointing=pointing,
                        instrument_type=instrument,
                    )

    # Plot covariance data and calibration history even if 1C01 file is not available
    elif prod == "1C01" and not os.path.isfile(input_file):
        output_dir = params["path_to_cal"]
        cov_data = prepare_data(
            "", "cov", params, None, date=time.mktime(date.timetuple())
        )
        assert isinstance(cov_data, dict)
        if len(cov_data) > 0 and output_dir is not None:
            generate_figure(
                "",
                ["tb_cov_ln2", "tb_cov_amb"],
                save_path=output_dir + "COVARIANCE/",
                instrument_type=instrument,
                image_name="cov",
                cov_data=cov_data,
                site=site,
            )
        else:
            logging.warning("Nothing to plot for product " + prod)
        his_data = prepare_data(
            "", "his", params, None, date=time.mktime(date.timetuple())
        )
        assert isinstance(his_data, dict)
        if len(his_data) > 0 and output_dir is not None:
            generate_figure(
                "",
                ["tb_cov_ln2", "tb_cov_amb", "Gain"]
                if "tb_cov_ln2" in his_data
                else ["Gain"],
                save_path=output_dir,
                instrument_type=instrument,
                image_name="his",
                cov_data=his_data,
                site=site,
            )
        else:
            logging.warning("Nothing to plot for product " + prod)
    else:
        logging.warning("Nothing to plot for product " + prod)


def _get_raw_file_path(
    date_in: datetime.date, site: str | None, instrument: str | None
) -> str:
    """Get the raw file path for a given date and site.

    Args:
        date_in: Date for which the raw file path is needed.
        site: Site identifier.
        instrument: Instrument identifier.

    Returns:
        The raw file path as a string.
    """
    params = mwrpy.utils.read_config(site, instrument, "params")
    return os.path.join(params["data_in"], date_in.strftime("%Y/%m/%d/"))


def _get_lidar_file_path(date_in: datetime.date, site: str, params: dict) -> str | None:
    """Get the lidar file path for a given date and site.

    Args:
        date_in: Date for which the lidar file path is needed.
        site: Site identifier.
        params: Configuration parameters.

    Returns:
        The lidar file path as a string or None if not found.
    """
    path = ""
    lidar_model = params.get("lidar_model", "unknown")
    lidar_model = "unknown" if lidar_model is None else lidar_model.lower()
    if "path_to_lidar" in params and params["path_to_lidar"] is not None:
        path = os.path.join(
            params["path_to_lidar"],
            date_in.strftime("%Y/%m/%d/"),
        )
    file = glob.glob(
        path + date_in.strftime("%Y%m%d") + "_" + site + "_" + lidar_model + "*.nc"
    )
    if len(file) == 0:
        logging.info(
            "No lidar file of type " + lidar_model + " found in directory " + str(path)
        )
        return None
    return file[0]
