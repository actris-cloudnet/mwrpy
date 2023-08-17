from tempfile import NamedTemporaryFile

import netCDF4

from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.utils import copy_global, copy_variables


def generate_lev2_single(
    site: str,
    mwr_l1c_file: str,
    output_file: str,
):
    with (
        NamedTemporaryFile() as lwp_file,
        NamedTemporaryFile() as iwv_file,
        NamedTemporaryFile() as t_prof_file,
        NamedTemporaryFile() as abs_hum_file,
    ):
        for prod, file in zip(
            ("2I01", "2I02", "2P01", "2P03"),
            (lwp_file, iwv_file, t_prof_file, abs_hum_file),
        ):
            lev2_to_nc(prod, mwr_l1c_file, site=site, output_file=file.name)

        with (
            netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
            netCDF4.Dataset(lwp_file.name, "r") as nc_lwp,
            netCDF4.Dataset(iwv_file.name, "r") as nc_iwv,
            netCDF4.Dataset(abs_hum_file.name, "r") as nc_hum,
            netCDF4.Dataset(t_prof_file.name, "r") as nc_t_prof,
        ):
            nc_output.createDimension("height", len(nc_t_prof.variables["height"][:]))
            nc_output.createDimension("time", len(nc_lwp.variables["time"][:]))
            nc_output.createDimension("bnds", 2)

            for source, variables in (
                (nc_iwv, ("iwv", "iwv_random_error", "iwv_systematic_error")),
                (
                    nc_hum,
                    (
                        "absolute_humidity",
                        "absolute_humidity_random_error",
                        "absolute_humidity_systematic_error",
                    ),
                ),
                (
                    nc_t_prof,
                    (
                        "temperature",
                        "temperature_random_error",
                        "temperature_systematic_error",
                        "height",
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
                    ),
                ),
            ):
                copy_variables(source, nc_output, variables)

            copy_global(nc_lwp, nc_output, nc_lwp.ncattrs())

        return nc_output


def generate_lev2_multi(site: str, mwr_l1c_file: str, output_file: str):
    with (
        NamedTemporaryFile() as temp_file,
        NamedTemporaryFile() as abs_hum_file,
        NamedTemporaryFile() as rel_hum_file,
        NamedTemporaryFile() as t_pot_file,
        NamedTemporaryFile() as eq_temp_file,
    ):
        for prod, file in zip(
            ("2P02", "2P03", "2P04", "2P07", "2P08"),
            (temp_file, abs_hum_file, rel_hum_file, t_pot_file, eq_temp_file),
        ):
            lev2_to_nc(
                prod,
                mwr_l1c_file,
                site=site,
                output_file=file.name,
                temp_file=temp_file.name if prod not in ("2P02", "2P03") else None,
                hum_file=abs_hum_file.name if prod not in ("2P02", "2P03") else None,
            )

        with (
            netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
            netCDF4.Dataset(temp_file.name, "r") as nc_temp,
            netCDF4.Dataset(rel_hum_file.name, "r") as nc_rel_hum,
            netCDF4.Dataset(t_pot_file.name, "r") as nc_t_pot,
            netCDF4.Dataset(eq_temp_file.name, "r") as nc_eq_temp,
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

        return nc_output
