"""Module for processing."""

import datetime
import glob
import logging
import os
import time

import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd

from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.level2.lev2_collocated import (
    generate_lev2_lhumpro,
    generate_lev2_multi,
    generate_lev2_single,
)
from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.plots.generate_plots import generate_figure
from mwrpy.utils import (
    _get_filename,
    _read_site_config_yaml,
    date_range,
    get_processing_dates,
    isodate2date,
    read_config,
)

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
            start = time.process_time()
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
                    process_product(product, date, args.site)
            if args.command != "no-plot":
                logging.info(f"Plotting {product} product, {args.site} {date}")
                try:
                    plot_product(product, date, args.site)
                except Exception as e:
                    logging.error(f"Error in plotting product {product}: {e}.")
                finally:
                    plt.close()
            elapsed_time = time.process_time() - start
            logging.info(f"Processing took {elapsed_time:.1f} seconds")


def process_product(prod: str, date: datetime.date, site: str):
    output_file = _get_filename(prod, date, site)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    lwp_offset: list[float | None] = [None, None]
    for iday in range(3):
        xday = [
            date - datetime.timedelta(days=iday + 1),
            date + datetime.timedelta(days=iday + 1),
        ]
        offset_file = [
            _get_filename("lwp_offset", xday[0], site),
            _get_filename("lwp_offset", xday[1], site),
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

    itype = _read_site_config_yaml(site)["type"]
    if prod[0] == "1":
        lev1_to_nc(
            prod,
            _get_raw_file_path(date, site),
            site=site,
            output_file=output_file,
            lidar_path=_get_lidar_file_path(date, site),
            date=date,
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
            lwp_offset=lwp_offset,
        )
    elif prod == "single" and itype != "lhumpro_u90":
        generate_lev2_single(
            site, _get_filename("1C01", date, site), output_file, lwp_offset
        )
    elif itype == "lhumpro_u90":
        generate_lev2_lhumpro(
            site, _get_filename("1C01", date, site), output_file, lwp_offset
        )
    elif prod == "multi":
        generate_lev2_multi(site, _get_filename("1C01", date, site), output_file)

    offset_current = _get_filename("lwp_offset", date, site)
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
            generate_figure(
                filename,
                variables,
                ele_range=ele_range,
                save_path=output_dir,
                image_name=key,
            )

    elif os.path.isfile(filename) and (prod[0] == "2"):
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
                    filename,
                    f_names,
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=PRODUCT_NAME[prod][0],
                    title=False,
                )
            elif key in ("lwp_scan", "iwv_scan"):
                generate_figure(
                    filename,
                    [key.rstrip("_scan")],
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=key,
                    title=False,
                )
            else:
                generate_figure(
                    filename,
                    [key],
                    ele_range=elevation,
                    save_path=output_dir,
                    image_name=key,
                    pointing=pointing,
                )

    elif os.path.isfile(filename) and (prod in ("single", "multi")):
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
                        filename,
                        [key.rstrip("_scan")],
                        ele_range=elevation,
                        save_path=output_dir,
                        image_name=key,
                        title=False,
                    )
                else:
                    generate_figure(
                        filename,
                        variables,
                        ele_range=elevation,
                        save_path=output_dir,
                        image_name=key,
                        title=title,
                        pointing=pointing,
                    )


def _get_raw_file_path(date_in: datetime.date, site: str) -> str:
    params = read_config(site, None, "params")
    return os.path.join(params["data_in"], date_in.strftime("%Y/%m/%d/"))


def _get_lidar_file_path(date_in: datetime.date, site: str) -> str | None:
    params, path = read_config(site, None, "params"), ""
    lidar_model = params["lidar_model"] if "lidar_model" in params else "unknown"
    if "path_to_lidar" in params and params["path_to_lidar"] is not None:
        path = os.path.join(
            params["path_to_lidar"],
            date_in.strftime("%Y/%m/%d/"),
            date_in.strftime("%Y%m%d") + "_" + site + "_" + lidar_model,
        )
    file = glob.glob(path + "*.nc")
    if len(file) == 0:
        logging.info(
            "No lidar file of type " + lidar_model + " found in directory " + str(path)
        )
        return None
    return file[0]
