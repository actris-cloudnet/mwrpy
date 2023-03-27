"""Module for Level 2 Metadata"""
from collections import namedtuple


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
        raise RuntimeError(["Data type " + data_type + " not supported for file writing."])

    else:

        attributes = dict(ATTRIBUTES_COM, **eval("ATTRIBUTES_" + data_type))
        for key in list(rpg_variables):
            if key in attributes:
                rpg_variables[key].set_attributes(attributes[key])
            else:
                del rpg_variables[key]

        index_map = {v: i for i, v in enumerate(attributes)}
        rpg_variables = dict(sorted(rpg_variables.items(), key=lambda pair: index_map[pair[0]]))

        return rpg_variables


FIELDS = ("long_name", "standard_name", "units", "definition", "comment")

MetaData = namedtuple("MetaData", FIELDS)
MetaData.__new__.__defaults__ = (None,) * len(MetaData._fields)


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
    "station_latitude": MetaData(
        long_name="Latitude of measurement station",
        standard_name="latitude",
        units="degrees_north",
    ),
    "station_longitude": MetaData(
        long_name="Longitude of measurement station",
        standard_name="longitude",
        units="degrees_east",
    ),
    "station_altitude": MetaData(
        long_name="Altitude above mean sea level of measurement station",
        standard_name="altitude",
        units="m",
    ),
    "azimuth_angle": MetaData(
        long_name="Sensor azimuth angle",
        standard_name="sensor_azimuth_angle",
        units="degrees",
        comment="0=North, 90=East, 180=South, 270=West",
    ),
    "ele": MetaData(
        long_name="Sensor elevation angle",
        standard_name="sensor_elevation_angle",
        units="degrees",
        comment="0=horizon, 90=zenith",
    ),
}


ATTRIBUTES_2P01 = {
    "altitude": MetaData(
        long_name="Altitude above sea level",
        units="m",
    ),
    "temperature": MetaData(
        long_name="Retrieved temperature profile (single pointing)",
        units="K",
    ),
    "temperature_random_error": MetaData(
        long_name="Random uncertainty of retrieved temperature profile (single pointing)",
        units="K",
        comment="specify here source of this variable",
    ),
    "temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved temperature profile (single pointing)",
        units="K",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2P02 = {
    "altitude": MetaData(
        long_name="Altitude above sea level",
        units="m",
    ),
    "temperature": MetaData(
        long_name="Retrieved temperature profile (multiple pointing)",
        units="K",
    ),
    "temperature_random_error": MetaData(
        long_name="Random uncertainty of retrieved temperature profile (multiple pointing)",
        units="K",
        comment="specify here source of this variable",
    ),
    "temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved temperature profile (multiple pointing)",
        units="K",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2P03 = {
    "altitude": MetaData(
        long_name="Altitude above sea level",
        units="m",
    ),
    "water_vapor_vmr": MetaData(
        long_name="Retrieved water vapour profile",
        units="ppm",
    ),
    "water_vapor_vmr_random_error": MetaData(
        long_name="Random uncertainty of retrieved water vapour profile",
        units="ppm",
        comment="specify here source of this variable",
    ),
    "water_vapor_vmr_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved water vapour profile",
        units="ppm",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2P04 = {
    "altitude": MetaData(
        long_name="Altitude above sea level",
        units="m",
    ),
    "relative_humidity": MetaData(
        long_name="Relative Humidity",
        units="1",
    ),
    "relative_humidity_random_error": MetaData(
        long_name="Random uncertainty of relative humidity",
        units="1",
        comment="specify here source of this variable",
    ),
    "relative_humidity_systematic_error": MetaData(
        long_name="Systematic uncertainty of relative humidity",
        units="1",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2P07 = {
    "altitude": MetaData(
        long_name="Altitude above sea level",
        units="m",
    ),
    "potential_temperature": MetaData(
        long_name="Potential Temperature",
        units="K",
    ),
    "potential_temperature_random_error": MetaData(
        long_name="Random uncertainty of potential temperature",
        units="K",
        comment="specify here source of this variable",
    ),
    "potential_temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of potential temperature",
        units="K",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2P08 = {
    "altitude": MetaData(
        long_name="Altitude above sea level",
        units="m",
    ),
    "equivalent_potential_temperature": MetaData(
        long_name="Equivalent Potential Temperature",
        units="K",
    ),
    "equivalent_potential_temperature_random_error": MetaData(
        long_name="Random uncertainty of equivalent potential temperature",
        units="K",
        comment="specify here source of this variable",
    ),
    "equivalent_potential_temperature_systematic_error": MetaData(
        long_name="Systematic uncertainty of equivalent potential temperature",
        units="K",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2I01 = {
    "lwp": MetaData(
        long_name="Retrieved column-integrated liquid water path",
        units="mm",
    ),
    "lwp_random_error": MetaData(
        long_name="Random uncertainty of retrieved column-integrated liquid water path",
        units="mm",
        comment="specify here source of this variable",
    ),
    "lwp_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved column-integrated liquid water path",
        units="mm",
        comment="specify here source of this variable",
    ),
    "lwp_offset": MetaData(
        long_name="Subtracted offset correction of retrieved column-integrated liquid water path",
        units="mm",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2I02 = {
    "iwv": MetaData(
        long_name="Retrieved column-integrated water vapour",
        units="mm",
    ),
    "iwv_random_error": MetaData(
        long_name="Random uncertainty of retrieved column-integrated water vapour",
        units="mm",
        comment="specify here source of this variable",
    ),
    "iwv_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved column-integrated water vapour",
        units="mm",
        comment="specify here source of this variable",
    ),
}


ATTRIBUTES_2S02 = {
    "frequency": MetaData(
        long_name="Centre frequency of microwave channels",
        standard_name="radiation_frequency",
        units="GHz",
        comment="For more accurate frequency values use frequency+freq_shift.",
    ),
    "receiver_nb": MetaData(
        long_name="Number of the microwave receiver",
    ),
    "receiver": MetaData(
        long_name="Corresponding microwave receiver for each channel",
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
