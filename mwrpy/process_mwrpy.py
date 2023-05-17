"""Module for processing."""
import datetime
import logging
import os

from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.plots.generate_plots import generate_figure
from mwrpy.utils import date_range, get_processing_dates, isodate2date, read_yaml_config

product = [
    ("1C01", ""),
    ("2I01", "lwp"),
    ("2I02", "iwv"),
    ("2P01", "temperature"),
    ("2P02", "temperature"),
    ("2P03", "absolute_humidity"),
    ("2P04", "relative_humidity"),
    ("2P07", "potential_temperature"),
    ("2P08", "equivalent_potential_temperature"),
    ("2S02", "tb_spectrum"),
    ("stats", ""),
]


def main(args):
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    for date in date_range(start_date, stop_date):
        for product in args.products:
            if args.figure:
                logging.info(f"Plotting {product} product, {args.site} {date}")
                plot_product(product, date, args.site)
            else:
                logging.info(f"Processing {product} product, {args.site} {date}")
                process_product(product, date, args.site)
                # plot_product(product, date, args.site)


def process_product(prod: str, date: datetime.date, site: str):
    output_file = get_filename(prod, date, site)
    output_dir = os.path.dirname(output_file)
    path_to_level1_files = output_dir.replace("/level2", "").replace("/level1", "")

    # Level 1
    if prod[0] == "1":
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        lev1_to_nc(
            site,
            prod,
            path_to_level1_files,
            output_file,
        )

    # Level 2
    elif (
        (prod[0] == "2")
        & (os.path.isdir("mwrpy/site_config/" + site + "/coefficients/"))
        & (os.path.isfile(get_filename("1C01", date, site)))
    ):
        path_to_level2_files = output_dir.replace("/level1", "/level2")

        if not os.path.isdir(path_to_level2_files):
            os.makedirs(path_to_level2_files)

        lev1_file = get_filename("1C01", date, site)

        if prod in ("2P04", "2P07", "2P08"):
            temp_file = get_filename("2P01", date, site)
            hum_file = get_filename("2P03", date, site)
        else:
            temp_file = None
            hum_file = None

        lev2_to_nc(
            site,
            prod,
            lev1_file,
            output_file,
            temp_file=temp_file,
            hum_file=hum_file,
        )


def get_filename(prod: str, date_in: datetime.date, site: str) -> str:
    global_attributes, params = read_yaml_config(site)
    data_out_dir = os.path.join(
        params["data_out"], f"level{prod[0]}", date_in.strftime("%Y/%m/%d")
    )
    wigos_id = global_attributes["wigos_station_id"]
    filename = f"MWR_{prod}_{wigos_id}_{date_in.strftime('%Y%m%d')}.nc"
    return os.path.join(data_out_dir, filename)


def plot_product(prod: str, date, site: str):
    global_attributes, params = read_yaml_config(site)
    ID = global_attributes["wigos_station_id"]
    data_out_l1 = params["data_out"] + "level1/" + date.strftime("%Y/%m/%d/")

    # Level 1
    if prod[0] == "1":
        lev1_data = (
            data_out_l1
            + "MWR_"
            + prod
            + "_"
            + ID
            + "_"
            + date.strftime("%Y%m%d")
            + ".nc"
        )
        if os.path.isfile(lev1_data):
            generate_figure(
                lev1_data,
                ["tb"],
                ele_range=[89.0, 91.0],
                save_path=data_out_l1,
                image_name="tb",
            )
            generate_figure(
                lev1_data,
                ["elevation_angle", "azimuth_angle"],
                save_path=data_out_l1,
                image_name="sen",
            )
            generate_figure(
                lev1_data,
                ["quality_flag"],
                save_path=data_out_l1,
                image_name="quality_flag",
            )
            generate_figure(
                lev1_data,
                ["met_quality_flag"],
                save_path=data_out_l1,
                image_name="met_quality_flag",
            )
            generate_figure(
                lev1_data,
                ["t_amb", "t_rec", "t_sta"],
                save_path=data_out_l1,
                image_name="hkd",
            )
            generate_figure(
                lev1_data,
                [
                    "air_temperature",
                    "relative_humidity",
                    "rainfall_rate",
                ],
                save_path=data_out_l1,
                image_name="met",
            )
            generate_figure(
                lev1_data,
                [
                    "air_pressure",
                    "wind_direction",
                    "wind_speed",
                ],
                save_path=data_out_l1,
                image_name="met2",
            )
            if params["ir_flag"]:
                generate_figure(
                    lev1_data,
                    ["irt"],
                    ele_range=[89.0, 91.0],
                    save_path=data_out_l1,
                    image_name="irt",
                )

    # Level 2
    elif (
        (prod[0] == "2")
        & (os.path.isdir("site_config/" + site + "/coefficients/"))
        & (
            os.path.isfile(
                data_out_l1 + "MWR_1C01_" + ID + "_" + date.strftime("%Y%m%d") + ".nc"
            )
        )
    ):
        data_out_l2 = params["data_out"] + "level2/" + date.strftime("%Y/%m/%d/")

        if prod in ("2I01", "2I02"):
            elevation = [89.0, 91.0]
        else:
            elevation = [0.0, 91.0]
        if os.path.isfile(
            data_out_l2
            + "MWR_"
            + prod
            + "_"
            + ID
            + "_"
            + date.strftime("%Y%m%d")
            + ".nc"
        ):
            var = dict(product)[prod]
            generate_figure(
                data_out_l2
                + "MWR_"
                + prod
                + "_"
                + ID
                + "_"
                + date.strftime("%Y%m%d")
                + ".nc",
                [var],
                ele_range=elevation,
                save_path=data_out_l2,
                image_name=var,
            )


def add_arguments(subparser):
    parser = subparser.add_parser("process", help="Process MWR Level 1 and 2 data.")
    parser.add_argument(
        "-f",
        "--figure",
        action="store_true",
        help="Produce figures only; no processing",
        default=False,
    )
    # parser.add_argument("indir")
    # parser.add_argument("outfile")
    # parser.add_argument("temp_file")
    # parser.add_argument("hum_file")
    return subparser
