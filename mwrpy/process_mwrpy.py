"""Module for processing."""
import datetime
import logging
import os

from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.plots.generate_plots import generate_figure
from mwrpy.utils import date_range, get_processing_dates, isodate2date, read_yaml_config

PRODUCT_NAME = {
    "1C01": "",
    "2I01": "lwp",
    "2I02": "iwv",
    "2P01": "temperature",
    "2P02": "temperature",
    "2P03": "absolute_humidity",
    "2P04": "relative_humidity",
    "2P07": "potential_temperature",
    "2P08": "equivalent_potential_temperature",
    "2S02": "tb_spectrum",
    "stats": "",
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
                process_product(product, date, args.site)
            logging.info(f"Plotting {product} product, {args.site} {date}")
            try:
                plot_product(product, date, args.site)
            except TypeError as err:
                logging.error(err)


def process_product(prod: str, date: datetime.date, site: str):
    output_file = _get_filename(prod, date, site)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if prod[0] == "1":
        lev1_to_nc(
            site,
            prod,
            _get_raw_file_path(date, site),
            output_file,
        )
    elif prod[0] == "2":
        if prod in ("2P04", "2P07", "2P08"):
            temp_file = _get_filename("2P01", date, site)
            hum_file = _get_filename("2P03", date, site)
        else:
            temp_file = None
            hum_file = None
        lev2_to_nc(
            site,
            prod,
            _get_filename("1C01", date, site),
            output_file,
            temp_file=temp_file,
            hum_file=hum_file,
        )


def plot_product(prod: str, date, site: str):
    filename = _get_filename(prod, date, site)
    output_dir = f"{os.path.dirname(filename)}/"

    if os.path.isfile(filename) and prod[0] == "1":
        keymap = {
            "tb": ["tb"],
            "sen": ["elevation_angle", "azimuth_angle"],
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
                if key == "tb"
                else (0.0, 91.0)
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
            if prod in ("2I01", "2I02")
            else (
                0,
                91.0,
            )
        )
        var = PRODUCT_NAME[prod]
        generate_figure(
            filename,
            [var],
            ele_range=elevation,
            save_path=output_dir,
            image_name=var,
        )


def _get_filename(prod: str, date_in: datetime.date, site: str) -> str:
    global_attributes, params = read_yaml_config(site)
    data_out_dir = os.path.join(
        params["data_out"], f"level{prod[0]}", date_in.strftime("%Y/%m/%d")
    )
    wigos_id = global_attributes["wigos_station_id"]
    filename = f"MWR_{prod}_{wigos_id}_{date_in.strftime('%Y%m%d')}.nc"
    return os.path.join(data_out_dir, filename)


def _get_raw_file_path(date_in: datetime.date, site: str) -> str:
    _, params = read_yaml_config(site)
    return os.path.join(params["data_in"], date_in.strftime("%Y/%m/%d/"))
