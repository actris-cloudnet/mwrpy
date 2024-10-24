"""Module for Level 2 Metadata."""

from collections.abc import Callable
from typing import TypeAlias

from mwrpy.utils import MetaData


def get_data_attributes(rpg_variables: dict, data_type: str, coeff: dict) -> dict:
    """Adds Metadata for RPG MWR Level 2 variables for NetCDF file writing.

    Args:
        rpg_variables: RpgArray instances.
        data_type: Data type of the netCDF file.
        coeff: Coefficient data of variable

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
        "2I06",
    ):
        raise RuntimeError(
            ["Data type " + data_type + " not supported for file writing."]
        )

    fields = [
        "retrieval_type",
        "retrieval_elevation_angles",
        "retrieval_frequencies",
        "retrieval_auxiliary_input",
        "retrieval_description",
    ]

    read_att = att_reader[data_type]
    attributes = dict(ATTRIBUTES_COM, **read_att)
    for key in list(rpg_variables):
        if key in attributes:
            if getattr(attributes[key], "retrieval_type") is not None:
                for field in fields:
                    if field in coeff:
                        attributes[key] = attributes[key]._replace(
                            **{field: coeff[field]}
                        )
            rpg_variables[key].set_attributes(attributes[key])
        else:
            del rpg_variables[key]

    index_map = {v: i for i, v in enumerate(attributes)}
    rpg_variables = dict(
        sorted(rpg_variables.items(), key=lambda pair: index_map[pair[0]])
    )

    return rpg_variables


DEFINITIONS_COM = {
    "quality_flag": (
        "\n"
        "Bit 1: missing_tb\n"
        "Bit 2: tb_below_threshold\n"
        "Bit 3: tb_above_threshold\n"
        "Bit 4: spectral_consistency_above_threshold\n"
        "Bit 5: receiver_sanity_failed\n"
        "Bit 6: rain_detected\n"
        "Bit 7: sun_moon_in_beam\n"
        "Bit 8: tb_offset_above_threshold"
    ),
    "quality_flag_status": (
        "\n"
        "Bit 1: missing_tb_not_checked\n"
        "Bit 2: tb_lower_threshold_not_checked\n"
        "Bit 3: tb_upper_threshold_not_checked\n"
        "Bit 4: spectral_consistency_not_checked\n"
        "Bit 5: receiver_sanity_not_checked\n"
        "Bit 6: rain_not_checked\n"
        "Bit 7: sun_moon_in_beam_not_checked\n"
        "Bit 8: tb_offset_not_checked"
    ),
}


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
        long_name="Azimuth angle",
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
        long_name="Temperature",
        comment="Retrieved temperature profile from single pointing measurements",
        standard_name="air_temperature",
        units="K",
        retrieval_type="",
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
    "temperature_quality_flag": MetaData(
        long_name="Quality flag",
        units="1",
        definition=DEFINITIONS_COM["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "temperature_quality_flag_status": MetaData(
        long_name="Quality flag status",
        units="1",
        definition=DEFINITIONS_COM["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
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
        comment="Retrieved temperature profile from multiple pointing measurements",
        standard_name="air_temperature",
        units="K",
        retrieval_type="",
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
    "temperature_quality_flag": MetaData(
        long_name="Quality flag",
        units="1",
        definition=DEFINITIONS_COM["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "temperature_quality_flag_status": MetaData(
        long_name="Quality flag status",
        units="1",
        definition=DEFINITIONS_COM["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
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
        retrieval_type="",
    ),
    "absolute_humidity_random_error": MetaData(
        long_name="Random uncertainty of absolute humidity",
        units="kg m-3",
    ),
    "absolute_humidity_systematic_error": MetaData(
        long_name="Systematic uncertainty of absolute humidity",
        units="kg m-3",
    ),
    "absolute_humidity_quality_flag": MetaData(
        long_name="Quality flag",
        units="1",
        definition=DEFINITIONS_COM["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "absolute_humidity_quality_flag_status": MetaData(
        long_name="Quality flag status",
        units="1",
        definition=DEFINITIONS_COM["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
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
        retrieval_type="",
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
        retrieval_type="",
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
        retrieval_type="",
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
        retrieval_type="",
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
    "lwp_quality_flag": MetaData(
        long_name="Quality flag",
        units="1",
        definition=DEFINITIONS_COM["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "lwp_quality_flag_status": MetaData(
        long_name="Quality flag status",
        units="1",
        definition=DEFINITIONS_COM["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
    ),
}


ATTRIBUTES_2I02 = {
    "iwv": MetaData(
        long_name="Retrieved column-integrated water vapour",
        standard_name="atmosphere_mass_content_of_water_vapor",
        units="kg m-2",
        retrieval_type="",
    ),
    "iwv_random_error": MetaData(
        long_name="Random uncertainty of retrieved column-integrated water vapour",
        units="kg m-2",
    ),
    "iwv_systematic_error": MetaData(
        long_name="Systematic uncertainty of retrieved column-integrated water vapour",
        units="kg m-2",
    ),
    "iwv_quality_flag": MetaData(
        long_name="Quality flag",
        units="1",
        definition=DEFINITIONS_COM["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "iwv_quality_flag_status": MetaData(
        long_name="Quality flag status",
        units="1",
        definition=DEFINITIONS_COM["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
    ),
}

ATTRIBUTES_2I06 = {
    "lifted_index": MetaData(
        long_name="Lifted index",
        units="1",
        retrieval_type="",
    ),
    "ko_index": MetaData(
        long_name="KO index",
        units="1",
        retrieval_type="",
    ),
    "total_totals": MetaData(
        long_name="Total totals index",
        units="1",
        retrieval_type="",
    ),
    "k_index": MetaData(
        long_name="K index",
        units="1",
        retrieval_type="",
    ),
    "showalter_index": MetaData(
        long_name="Showalter index",
        units="1",
        retrieval_type="",
    ),
    "cape": MetaData(
        long_name="Convective available potential energy",
        units="1",
        retrieval_type="",
    ),
    "stability_quality_flag": MetaData(
        long_name="Quality flag for stability products",
        units="1",
        definition=DEFINITIONS_COM["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "stability_quality_flag_status": MetaData(
        long_name="Quality flag status for stability products",
        units="1",
        definition=DEFINITIONS_COM["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
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
    "2I06": ATTRIBUTES_2I06,
}
