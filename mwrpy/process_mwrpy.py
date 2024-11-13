"""Module for processing."""

import datetime
import logging
import os
import time

import matplotlib.pyplot as plt

from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.level2.lev2_collocated import generate_lev2_multi, generate_lev2_single
from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.plots.generate_plots import generate_figure
from mwrpy.utils import (
    _get_filename,
    date_range,
    get_processing_dates,
    isodate2date,
    read_config,
)

PRODUCT_NAME = {
    "1B01": "",
    "1B11": "",
    "1B21": "",
    "1C01": "",
    "2I01": "lwp",
    "2I02": "iwv",
    "2I06": "stability",
    "2P01": "temperature",
    "2P02": "temperature",
    "2P03": "absolute_humidity",
    "2P04": "relative_humidity",
    "2P07": "potential_temperature",
    "2P08": "equivalent_potential_temperature",
    "single": [
        "lwp",
        "iwv",
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


def main(args):
    logging.basicConfig(level="INFO")
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    for date in date_range(start_date, stop_date):
        for product in args.products:
            if product not in PRODUCT_NAME:
                logging.error(f"Product {product} not recognised")
                continue
            if args.command != "plot":
                logging.info(f"Processing {product} product, {args.site} {date}")
                if args.command == "reprocess":
                    try:
                        process_product(product, date, args.site)
                    except Exception as e:
                        logging.error(
                            f"Error in processing products: {e}. Incomplete or no processing for {date}."
                        )
                else:
                    start = time.process_time()
                    process_product(product, date, args.site)
                    elapsed_time = time.process_time() - start
                    logging.info(f"Processing took {elapsed_time:.1f} seconds")
            if args.command != "no-plot":
                logging.info(f"Plotting {product} product, {args.site} {date}")
                try:
                    plot_product(product, date, args.site)
                except TypeError as err:
                    logging.error(err)
                plt.close()


def process_product(prod: str, date: datetime.date, site: str):
    output_file = _get_filename(prod, date, site)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if prod[0] == "1":
        lev1_to_nc(
            prod,
            _get_raw_file_path(date, site),
            site=site,
            output_file=output_file,
        )
    elif prod[0] == "2":
        if prod in ("2P04", "2P07", "2P08"):
            temp_file = _get_filename("2P02", date, site)
            if len(temp_file) == 0:
                temp_file = _get_filename("2P01", date, site)
            hum_file = _get_filename("2P03", date, site)
        else:
            temp_file = None
            hum_file = None
        lev2_to_nc(
            prod,
            _get_filename("1C01", date, site),
            output_file=output_file,
            site=site,
            temp_file=temp_file,
            hum_file=hum_file,
        )
    elif prod == "single":
        generate_lev2_single(site, _get_filename("1C01", date, site), output_file)
    elif prod == "multi":
        generate_lev2_multi(site, _get_filename("1C01", date, site), output_file)


def plot_product(prod: str, date, site: str):
    filename = _get_filename(prod, date, site)
    if not os.path.isfile(filename):
        logging.warning("Nothing to plot for product " + prod)
    output_dir = f"{os.path.dirname(filename)}/"

    if os.path.isfile(filename) and prod[0] == "1":
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
        }
        for key, variables in keymap.items():
            ele_range = (
                (
                    89.0,
                    91.0,
                )
                if key in ("tb", "tb_spectrum")
                else (-1.0, 91.0)
            )
            generate_figure(
                filename,
                variables,
                ele_range=ele_range,
                save_path=output_dir,
                image_name=key,
            )

    elif os.path.isfile(filename) and (prod[0] == "2"):
        elevation = (
            (
                89.0,
                91.0,
            )
            if prod not in ("2P02", "2P04", "2P07", "2P08")
            else (
                -1.0,
                91.0,
            )
        )
        if prod == "2I06":
            f_names = list(
                [
                    "cape",
                    "k_index",
                    "total_totals",
                    "lifted_index",
                    "showalter_index",
                    "ko_index",
                ]
            )
            generate_figure(
                filename,
                f_names,
                ele_range=elevation,
                save_path=output_dir,
                image_name=str(PRODUCT_NAME[prod]),
                title=False,
            )
        else:
            generate_figure(
                filename,
                [PRODUCT_NAME[prod]],
                ele_range=elevation,
                save_path=output_dir,
                image_name=str(PRODUCT_NAME[prod]),
            )

    elif os.path.isfile(filename) and (prod in ("single", "multi")):
        for var_name in PRODUCT_NAME[prod]:
            elevation = (
                (
                    89.0,
                    91.0,
                )
                if prod == "single"
                else (
                    -1.0,
                    91.0,
                )
            )
            f_names = [
                "cape",
                "k_index",
                "total_totals",
                "lifted_index",
                "showalter_index",
                "ko_index",
            ]
            if var_name == "stability":
                keymap = {
                    var_name: f_names,
                }
            else:
                keymap = {
                    var_name: [var_name],
                }
            title = False if var_name in f_names else True
            for key, variables in keymap.items():
                generate_figure(
                    filename,
                    variables,
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=key,
                    title=title,
                )


def _get_raw_file_path(date_in: datetime.date, site: str) -> str:
    params = read_config(site, "params")
    return os.path.join(params["data_in"], date_in.strftime("%Y/%m/%d/"))
