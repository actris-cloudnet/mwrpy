"""Module for processing."""
import os
import datetime
from .level1.write_lev1_nc import lev1_to_nc
from .level2.write_lev2_nc import lev2_to_nc
from .plots.generate_plots import generate_figure
from .plots.stat_plot import generate_stat
from .utils import (
    read_yaml_config,
    get_processing_dates,
    isodate2date,
    date_range,
)

product = [
    ("1C01", ""),
    ("2I01", "lwp"),
    ("2I02", "iwv"),
    ("2P01", "temperature"),
    ("2P02", "temperature"),
    ("2P03", "water_vapor_vmr"),
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
                print(f"Plotting {product} product, {args.site} {date}")
                plot_product(product, date, args.site)
            else:
                print(f"Processing {product} product, {args.site} {date}")
                process_product(product, date, args.site)
                print(f"Plotting {product} product, {args.site} {date}")
                plot_product(product, date, args.site)
                
            
def _link_quicklook(link_direc: str, figure_name: str) -> None:
    if len(figure_name) > 0:
        if not os.path.isdir(link_direc):
            os.makedirs(link_direc)


def process_product(
        prod: str,
        date: str | None,
        site: str,
        input_dir: str,
        output_filename: str) -> None:

    if prod[0] == "1":
        lev1_to_nc(
            site,
            prod,
            input_dir,
            output_filename,
        )

    # Level 2

    # global_attributes, params = read_yaml_config(site)
    # ID = global_attributes["wigos_station_id"]
    # elif (
    #     (prod[0] == "2")
    #     & (
    #         os.path.isdir("site_config/" + site + "/coefficients/")
    #     ) & (
    #         os.path.isfile(output_dir + "MWR_1C01_" + ID + "_" + date.strftime("%Y%m%d") + ".nc")
    #     )
    # ):
    #     data_out_l2 = params["data_out"] + "level2/" + date.strftime("%Y/%m/%d/")
    #     if not os.path.isdir(data_out_l2):
    #         os.makedirs(data_out_l2)
    #     lev1_data = output_dir + "MWR_1C01_" + ID + "_" + date.strftime("%Y%m%d") + ".nc"
    #     lev2_to_nc(date.strftime("%Y%m%d"), site, prod, lev1_data, data_out_l2)
        
        
def plot_product(prod: str, date, site: str):
    global_attributes, params = read_yaml_config(site)
    ID = global_attributes["wigos_station_id"]
    data_out_l1 = params["data_out"] + "level1/" + date.strftime("%Y/%m/%d/")

    # Level 1
    if prod[0] == "1":
        link_dir = (
            "/tmp/"
            + params["data_out"][6:]
            + "level1/"
            + date.strftime("%Y/%m/%d/")
        )
        
        lev1_data = data_out_l1 + "MWR_" + prod + "_" + ID + "_" + date.strftime("%Y%m%d") + ".nc"        
        if os.path.isfile(lev1_data):

            fig_name = generate_figure(
                lev1_data,
                ["tb"],
                ele_range=[89.0, 91.0],
                save_path=data_out_l1,
                image_name="tb",
            )
            _link_quicklook(link_dir, fig_name)
            fig_name = generate_figure(
                lev1_data, ["ele", "azimuth_angle"], save_path=data_out_l1, image_name="sen"
            )
            _link_quicklook(link_dir, fig_name)
            fig_name = generate_figure(
                lev1_data,
                ["quality_flag"],
                save_path=data_out_l1,
                image_name="quality_flag",
            )
            _link_quicklook(link_dir, fig_name)
            fig_name = generate_figure(
                lev1_data,
                ["met_quality_flag"],
                save_path=data_out_l1,
                image_name="met_quality_flag",
            )
            _link_quicklook(link_dir, fig_name)
            fig_name = generate_figure(
                lev1_data,
                ["t_amb", "t_rec", "t_sta"],
                save_path=data_out_l1,
                image_name="hkd",
            )
            _link_quicklook(link_dir, fig_name)
            fig_name = generate_figure(
                lev1_data,
                [
                    "air_temperature",
                    "relative_humidity",
                    "rainfall_rate",
                ],
                save_path=data_out_l1,
                image_name="met",
            )
            _link_quicklook(link_dir, fig_name)
            fig_name = generate_figure(
                lev1_data,
                [
                    "air_pressure",
                    "wind_direction",
                    "wind_speed",
                ],
                save_path=data_out_l1,
                image_name="met2",
            )
            _link_quicklook(link_dir, fig_name)
            if params["ir_flag"]:
                fig_name = generate_figure(
                    lev1_data,
                    ["irt"],
                    ele_range=[89.0, 91.0],
                    save_path=data_out_l1,
                    image_name="irt",
                )
                _link_quicklook(link_dir, fig_name)

    # Level 2
    elif (
        (prod[0] == "2") 
        & (
            os.path.isdir("site_config/" + site + "/coefficients/")
        ) & (
            os.path.isfile(data_out_l1 + "MWR_1C01_" + ID + "_" + date.strftime("%Y%m%d") + ".nc")
        )
    ):
        link_dir = (
            "/tmp/"
            + params["data_out"][6:]
            + "level2/"
            + date.strftime("%Y/%m/%d/")
        )
        data_out_l2 = params["data_out"] + "level2/" + date.strftime("%Y/%m/%d/")

        if prod in ("2I01", "2I02"):
            elevation = [89.0, 91.0]
        else:
            elevation = [0.0, 91.0]
        if os.path.isfile(
            data_out_l2 + "MWR_" + prod + "_" + ID + "_" + date.strftime("%Y%m%d") + ".nc"
        ):
            var = dict(product)[prod]
            fig_name = generate_figure(
                data_out_l2 + "MWR_" + prod + "_" + ID + "_" + date.strftime("%Y%m%d") + ".nc",
                [var],
                ele_range=elevation,
                save_path=data_out_l2,
                image_name=var,
            )
            _link_quicklook(link_dir, fig_name)
    
    # Statistics
    elif prod == "stats":
        data_out_stat = params["data_out"] + "level1/" + date.strftime("%Y/")
        if not os.path.isdir(data_out_stat):
            os.makedirs(data_out_stat)
        if params["flag_status"][3] == 1:
            generate_stat(
                site,
                ["data_availability", "quality_flag"],
                date.strftime("%Y"),
                "data_stat",
                data_out_stat,
            )
        else:
            generate_stat(
                site,
                ["data_availability", "quality_flag", "spectral_consistency"],
                date.strftime("%Y"),
                "data_stat",
                data_out_stat,
            )
        generate_stat(
            site,
            ["receiver_temperature", "receiver_stability"],
            date.strftime("%Y"),
            "receiver_stat",
            data_out_stat,
        )
        generate_stat(
            site,
            ["ambient_target"],
            date.strftime("%Y"),
            "ambient_target_stat",
            data_out_stat,
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
    return subparser
