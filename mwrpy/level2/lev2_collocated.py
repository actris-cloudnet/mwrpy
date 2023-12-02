import os

import netCDF4

from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.utils import copy_global, copy_variables


def generate_lev2_single(
    site: str | None,
    mwr_l1c_file: str,
    output_file: str,
    coeff_files: list[str] | None = None,
):
    products = {
        "2I01": "lwp",
        "2I02": "iwv",
        "2P01": "t_prof",
        "2P03": "abs_hum",
        "2P04": "rel_hum",
        "2P07": "t_pot",
        "2P08": "eq_temp",
    }

    for prod, file in products.items():
        lev2_to_nc(
            prod,
            mwr_l1c_file,
            output_file=file,
            site=site,
            temp_file=products["2P01"] if prod in ("2P04", "2P07", "2P08") else None,
            hum_file=products["2P03"] if prod in ("2P04", "2P07", "2P08") else None,
            coeff_files=coeff_files,
        )

    with (
        netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
        netCDF4.Dataset(products["2I01"], "r") as nc_lwp,
        netCDF4.Dataset(products["2I02"], "r") as nc_iwv,
        netCDF4.Dataset(products["2P01"], "r") as nc_t_prof,
        netCDF4.Dataset(products["2P03"], "r") as nc_abs_hum,
        netCDF4.Dataset(products["2P04"], "r") as nc_rel_hum,
        netCDF4.Dataset(products["2P07"], "r") as nc_t_pot,
        netCDF4.Dataset(products["2P08"], "r") as nc_eq_temp,
    ):
        nc_output.createDimension("height", len(nc_t_prof.variables["height"][:]))
        nc_output.createDimension("time", len(nc_lwp.variables["time"][:]))
        nc_output.createDimension("bnds", 2)

        for source, variables in (
            (
                nc_iwv,
                (
                    "iwv",
                    "iwv_random_error",
                    "iwv_systematic_error",
                    "iwv_quality_flag",
                    "iwv_quality_flag_status",
                ),
            ),
            (
                nc_abs_hum,
                (
                    "absolute_humidity",
                    "absolute_humidity_random_error",
                    "absolute_humidity_systematic_error",
                    "absolute_humidity_quality_flag",
                    "absolute_humidity_quality_flag_status",
                ),
            ),
            (
                nc_t_prof,
                (
                    "temperature",
                    "temperature_random_error",
                    "temperature_systematic_error",
                    "height",
                    "temperature_quality_flag",
                    "temperature_quality_flag_status",
                ),
            ),
            (
                nc_lwp,
                (
                    "time",
                    "time_bnds",
                    "latitude",
                    "longitude",
                    "altitude",
                    "lwp",
                    "lwp_offset",
                    "lwp_random_error",
                    "lwp_systematic_error",
                    "elevation_angle",
                    "azimuth_angle",
                    "lwp_quality_flag",
                    "lwp_quality_flag_status",
                ),
            ),
            (
                nc_rel_hum,
                (
                    "relative_humidity",
                    "relative_humidity_random_error",
                    "relative_humidity_systematic_error",
                ),
            ),
            (
                nc_t_pot,
                (
                    "potential_temperature",
                    "potential_temperature_random_error",
                    "potential_temperature_systematic_error",
                ),
            ),
            (
                nc_eq_temp,
                (
                    "equivalent_potential_temperature",
                    "equivalent_potential_temperature_random_error",
                    "equivalent_potential_temperature_systematic_error",
                ),
            ),
        ):
            copy_variables(source, nc_output, variables)

        copy_global(nc_lwp, nc_output, nc_lwp.ncattrs())

        _delete_temp_files(products)

        return nc_output


def generate_lev2_multi(
    site: str | None,
    mwr_l1c_file: str,
    output_file: str,
    coeff_files: list[str] | None = None,
):
    products = {
        "2P02": "temp",
        "2P03": "abs_hum",
        "2P04": "rel_hum",
        "2P07": "t_pot",
        "2P08": "eq_temp",
    }

    for prod, file in products.items():
        lev2_to_nc(
            prod,
            mwr_l1c_file,
            output_file=file,
            site=site,
            temp_file=products["2P02"] if prod not in ("2P02", "2P03") else None,
            hum_file=products["2P03"] if prod not in ("2P02", "2P03") else None,
            coeff_files=coeff_files,
        )

    with (
        netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
        netCDF4.Dataset(products["2P02"], "r") as nc_temp,
        netCDF4.Dataset(products["2P04"], "r") as nc_rel_hum,
        netCDF4.Dataset(products["2P07"], "r") as nc_t_pot,
        netCDF4.Dataset(products["2P08"], "r") as nc_eq_temp,
    ):
        nc_output.createDimension("time", len(nc_temp.variables["time"][:]))
        nc_output.createDimension("height", len(nc_temp.variables["height"][:]))
        nc_output.createDimension("bnds", 2)

        for source, variables in (
            (
                nc_temp,
                (
                    "time",
                    "time_bnds",
                    "height",
                    "latitude",
                    "longitude",
                    "altitude",
                    "elevation_angle",
                    "azimuth_angle",
                    "temperature",
                    "temperature_random_error",
                    "temperature_systematic_error",
                    "temperature_quality_flag",
                    "temperature_quality_flag_status",
                ),
            ),
            (
                nc_rel_hum,
                (
                    "relative_humidity",
                    "relative_humidity_random_error",
                    "relative_humidity_systematic_error",
                ),
            ),
            (
                nc_t_pot,
                (
                    "potential_temperature",
                    "potential_temperature_random_error",
                    "potential_temperature_systematic_error",
                ),
            ),
            (
                nc_eq_temp,
                (
                    "equivalent_potential_temperature",
                    "equivalent_potential_temperature_random_error",
                    "equivalent_potential_temperature_systematic_error",
                ),
            ),
        ):
            copy_variables(source, nc_output, variables)

        copy_global(nc_temp, nc_output, nc_temp.ncattrs())

        _delete_temp_files(products)

        return nc_output


def _delete_temp_files(products: dict):
    for _, file in products.items():
        if os.path.exists(file):
            try:
                os.remove(file)
            except PermissionError:
                pass
