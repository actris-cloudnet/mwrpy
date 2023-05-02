"""Module for Level 2 Metadata"""
from collections.abc import Callable
from typing import TypeAlias

from mwrpy.utils import MetaData


def get_data_attributes(rpg_variables: dict, data_type: str) -> dict:
    """Adds Metadata for RPG MWR Level 2 variables for NetCDF file writing.
    Args:
        rpg_variables: RpgArray instances.
        data_type: Data type of the netCDF file.

    Returns:
        Dictionary

    Raises:
        RuntimeError: Specified data type is not supported.

    Example:
        from level2.lev2_meta_nc import get_data_attributes
        att = get_data_attributes('data','data_type')
    """

    if data_type not in (
        "2P01",
        "2P02",
        "2P03",
        "2P04",
        "2P07",
        "2P08",
        "2I01",
        "2I02",
        "2S02",
    ):
        raise RuntimeError(
            ["Data type " + data_type + " not supported for file writing."]
        )

    read_att = att_reader[data_type]
    attributes = dict(ATTRIBUTES_COM, **read_att)
    for key in list(rpg_variables):
        if key in attributes:
            rpg_variables[key].set_attributes(attributes[key])
        else:
            del rpg_variables[key]

    index_map = {v: i for i, v in enumerate(attributes)}
    rpg_variables = dict(
        sorted(rpg_variables.items(), key=lambda pair: index_map[pair[0]])
    )

    return rpg_variables


ATTRIBUTES_COM = {
    "time": MetaData(
        long_name="Time (UTC) of the measurement",
        units="seconds since 1970-01-01 00:00:00.000",
        comment="Time indication of samples is at end of integration-time",
    ),
    "time_bnds": MetaData(
        long_name="Start and end time (UTC) of the measurements",
        units="seconds since 1970-01-01 00:00:00.000",
    ),
    "latitude": MetaData(
        long_name="Latitude of measurement station",
        standard_name="latitude",
        units="degree_north",
    ),
    "longitude": MetaData(
        long_name="Longitude of measurement station",
        standard_name="longitude",
        units="degree_east",
    ),
    "altitude": MetaData(
        long_name="Altitude above mean sea level of measurement station",
        standard_name="altitude",
        units="m",
    ),
    "azimuth_angle": MetaData(
        long_name="Sensor azimuth angle",
        standard_name="sensor_azimuth_angle",
        units="degree",
        comment="0=North, 90=East, 180=South, 270=West",
    ),
    "elevation_angle": MetaData(
        long_name="Sensor elevation angle",
        units="degree",
        comment="0=horizon, 90=zenith",
    ),
}


ATTRIBUTES_2P01 = {
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "temperature": MetaData(
        long_name="Retrieved temperature profile (single pointing)",
        standard_name="air_temperature",
        units="K",
    ),
    "temperature_random_error": MetaData(
        long_name="Random uncertainty of retrieved\n"
        "temperature profile (single pointing)",
        units="K",
    ),
    "temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved\n"
        "temperature profile (single pointing)",
        units="K",
    ),
}


ATTRIBUTES_2P02 = {
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "temperature": MetaData(
        long_name="Temperature",
        comment="Retrieved temperature profile from multiple pointing mwr measurements",
        standard_name="air_temperature",
        units="K",
    ),
    "temperature_random_error": MetaData(
        long_name="Random uncertainty of retrieved\n"
        "temperature profile (multiple pointing)",
        units="K",
    ),
    "temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved\n"
        "temperature profile (multiple pointing)",
        units="K",
    ),
}


ATTRIBUTES_2P03 = {
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "absolute_humidity": MetaData(
        long_name="Absolute humidity",
        units="kg m-3",
    ),
    "absolute_humidity_random_error": MetaData(
        long_name="Random uncertainty of absolute humidity",
        units="kg m-3",
    ),
    "absolute_humidity_systematic_error": MetaData(
        long_name="Systematic uncertainty of absolute humidity",
        units="kg m-3",
    ),
}


ATTRIBUTES_2P04 = {
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "relative_humidity": MetaData(
        long_name="Relative humidity",
        standard_name="relative_humidity",
        units="1",
    ),
    "relative_humidity_random_error": MetaData(
        long_name="Random uncertainty of relative humidity",
        units="1",
    ),
    "relative_humidity_systematic_error": MetaData(
        long_name="Systematic uncertainty of relative humidity",
        units="1",
    ),
}


ATTRIBUTES_2P07 = {
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "potential_temperature": MetaData(
        long_name="Potential temperature",
        standard_name="air_potential_temperature",
        units="K",
    ),
    "potential_temperature_random_error": MetaData(
        long_name="Random uncertainty of potential temperature",
        units="K",
    ),
    "potential_temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of potential temperature",
        units="K",
    ),
}


ATTRIBUTES_2P08 = {
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "equivalent_potential_temperature": MetaData(
        long_name="Equivalent potential temperature",
        standard_name="air_equivalent_potential_temperature",
        units="K",
    ),
    "equivalent_potential_temperature_random_error": MetaData(
        long_name="Random uncertainty of equivalent potential temperature",
        units="K",
    ),
    "equivalent_potential_temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of equivalent potential temperature",
        units="K",
    ),
}


ATTRIBUTES_2I01 = {
    "lwp": MetaData(
        long_name="Retrieved column-integrated liquid water path",
        standard_name="atmosphere_cloud_liquid_water_content",
        units="kg m-2",
    ),
    "lwp_random_error": MetaData(
        long_name="Random uncertainty of retrieved\n"
        "column-integrated liquid water path",
        units="kg m-2",
    ),
    "lwp_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved\n"
        "column-integrated liquid water path",
        units="kg m-2",
    ),
    "lwp_offset": MetaData(
        long_name="Subtracted offset correction of liquid water path",
        units="kg m-2",
    ),
}


ATTRIBUTES_2I02 = {
    "iwv": MetaData(
        long_name="Retrieved column-integrated water vapour",
        standard_name="atmosphere_mass_content_of_water_vapor",
        units="kg m-2",
    ),
    "iwv_random_error": MetaData(
        long_name="Random uncertainty of retrieved column-integrated water vapour",
        units="kg m-2",
    ),
    "iwv_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved column-integrated water vapour",
        units="kg m-2",
    ),
}


ATTRIBUTES_2S02 = {
    "frequency": MetaData(
        long_name="Centre frequency of microwave channels",
        standard_name="radiation_frequency",
        units="GHz",
        comment="For more accurate frequency values use frequency+freq_shift.",
    ),
    "receiver_nb": MetaData(long_name="Number of the microwave receiver", units="1"),
    "receiver": MetaData(
        long_name="Corresponding microwave receiver for each channel", units="1"
    ),
    "tb": MetaData(
        long_name="Microwave brightness temperatures",
        standard_name="microwave_brightness_temperature",
        units="K",
    ),
    "tb_spectrum": MetaData(
        long_name="Brightness temperature spectrum",
        units="K",
    ),
}

FuncType: TypeAlias = Callable[[str], dict]
att_reader: dict[str, dict] = {
    "2P01": ATTRIBUTES_2P01,
    "2P02": ATTRIBUTES_2P02,
    "2P03": ATTRIBUTES_2P03,
    "2P04": ATTRIBUTES_2P04,
    "2P07": ATTRIBUTES_2P07,
    "2P08": ATTRIBUTES_2P08,
    "2I01": ATTRIBUTES_2I01,
    "2I02": ATTRIBUTES_2I02,
    "2S02": ATTRIBUTES_2S02,
}
