"""Module for Level 1 Metadata."""

from collections.abc import Callable
from typing import TypeAlias

from mwrpy.utils import MetaData


def get_data_attributes(rpg_variables: dict, data_type: str) -> dict:
    """Adds Metadata for RPG MWR Level 1 variables for NetCDF file writing.

    Args:
        rpg_variables: RpgArray instances.
        data_type: Data type of the netCDF file.

    Returns:
        Dictionary

    Raises:
        RuntimeError: Specified data type is not supported.

    Example:
        from level1.lev1_meta_nc import get_data_attributes
        att = get_data_attributes('data','data_type')
    """
    if data_type not in (
        "1B01",
        "1B11",
        "1B21",
        "1C01",
    ):
        raise RuntimeError(
            ["Data type " + data_type + " not supported for file writing."]
        )

    if data_type in ("1B01", "1B11", "1B21"):
        read_att = att_reader[data_type]
        attributes = dict(ATTRIBUTES_COM, **read_att)

    elif data_type == "1C01":
        attributes = dict(
            ATTRIBUTES_COM, **ATTRIBUTES_1B01, **ATTRIBUTES_1B11, **ATTRIBUTES_1B21
        )

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
        units="degrees_north",
    ),
    "longitude": MetaData(
        long_name="Longitude of measurement station",
        standard_name="longitude",
        units="degrees_east",
    ),
    "altitude": MetaData(
        long_name="Altitude above mean sea level of measurement station",
        standard_name="altitude",
        units="m",
    ),
}


DEFINITIONS_1B01 = {
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

ATTRIBUTES_1B01 = {
    "frequency": MetaData(
        long_name="Nominal centre frequency of microwave channels",
        standard_name="radiation_frequency",
        units="GHz",
        comment="1) For double-sideband receivers, frequency corresponds to the\n"
        "local oscillator frequency whereas the radio frequency of the upper/lower\n"
        "sideband is frequency+/-sideband_IF_separation. 2) In case of known\n"
        "offset between the real and the nominal frequency of some channels,\n"
        "frequency+freq_shift gives more accurate values.",
    ),
    "receiver_nb": MetaData(
        long_name="Microwave receiver number",
        units="1",
    ),
    "receiver": MetaData(
        long_name="Corresponding microwave receiver for each channel", units="1"
    ),
    "bandwidth": MetaData(
        long_name="Bandwidth of microwave channels",
        units="GHz",
    ),
    "n_sidebands": MetaData(
        long_name="Number of sidebands",
        units="1",
        comment="0: direct-detection receivers, 1: single-sideband,\n"
        "2: double-sideband. The frequency separation of sidebands\n"
        "is indicated in sideband_IF_separation.",
    ),
    "sideband_IF_separation": MetaData(
        long_name="Sideband IF separation",
        comment="For double sideband channels, this is the positive and negative\n"
        "IF range distance of the two band passes around the centre frequency\n"
        "(which is the LO frqeuency)",
        units="GHz",
    ),
    "beamwidth": MetaData(
        long_name="Beam width (FWHM) of the microwave radiometer",
        units="degree",
    ),
    "freq_shift": MetaData(
        long_name="Frequency shift of the microwave channels",
        comment="For more accurate frequency values use frequency + freq_shift.",
        units="GHz",
    ),
    "tb": MetaData(
        long_name="Microwave brightness temperature",
        standard_name="brightness_temperature",
        units="K",
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
    "tb_accuracy": MetaData(
        long_name="Total absolute calibration uncertainty of brightness temperature,\n"
        "one standard deviation",
        units="K",
        comment="specify here source of this variable, e.g. literature value,\n"
        "specified by manufacturer, result of validation effort\n"
        "(updated irregularily) For RDX systems, derived from analysis\n"
        "performed by Tim Hewsion (Tim J. Hewison, 2006: Profiling Temperature\n"
        "and Humidity by Ground-based Microwave Radiometers, PhD Thesis,\n"
        "University of Reading.) Derived from sensitivity analysis of LN2\n"
        "calibration plus instrument noise levels (ACTRIS work), \n"
        "currently literature values (Maschwitz et al. for HATPRO, ? for radiometrics)",
    ),
    "tb_cov": MetaData(
        long_name="Error covariance matrix of brightness temperature channels",
        units="K*K",
        comment="the covariance matrix has been determined using the xxx method\n"
        "from observations at a blackbody target of temperature t_amb",
    ),
    "quality_flag": MetaData(
        long_name="Quality flag",
        units="1",
        definition=DEFINITIONS_1B01["quality_flag"],
        comment="0 indicates data with good quality according to applied tests.\n"
        "The list of (not) applied tests is encoded in quality_flag_status",
    ),
    "quality_flag_status": MetaData(
        long_name="Quality flag status",
        units="1",
        definition=DEFINITIONS_1B01["quality_flag_status"],
        comment="Checks not executed in determination of quality_flag.\n"
        "0 indicates quality check has been applied.",
    ),
    "liquid_cloud_flag": MetaData(
        long_name="Liquid cloud flag",
        units="1",
        comment="Flag meaning: no liquid cloud (0), liquid cloud present (1),\n"
        "undefined (2)",
    ),
    "liquid_cloud_flag_status": MetaData(
        long_name="Liquid cloud flag status",
        units="1",
        comment="Flag meaning: using mwr and ir (0), using mwr only (1), other (2)",
    ),
    "pointing_flag": MetaData(
        long_name="Pointing flag",
        units="1",
        comment="Flag indicating a single pointing (staring = 0)\n"
        "or multiple pointing (scanning = 1) observation sequence",
    ),
    "t_amb": MetaData(
        long_name="Ambient target temperature",
        units="K",
    ),
    "t_rec": MetaData(
        long_name="Receiver physical temperature",
        units="K",
    ),
    "t_sta": MetaData(
        long_name="Receiver temperature stability",
        units="K",
    ),
    "ir_zenith_angle": MetaData(
        long_name="Infrared sensor zenith angle",
        units="degree",
        comment="90=horizon, 0=zenith",
    ),
    "tb_spectrum": MetaData(
        long_name="Brightness temperature spectrum",
        units="K",
    ),
    # 'tn': MetaData(
    #     long_name='Receiver noise temperature',
    #     units='K',
    # )
}


ATTRIBUTES_1B11 = {
    "ir_wavelength": MetaData(
        long_name="Wavelength of infrared channels",
        standard_name="sensor_band_central_radiation_wavelength",
        units="m",
    ),
    "ir_bandwidth": MetaData(
        long_name="Bandwidth of infrared channels",
        units="m",
        comment="Channel centre frequency.",
    ),
    "ir_beamwidth": MetaData(
        long_name="Beam width of the infrared radiometer",
        units="degree",
    ),
    "irt": MetaData(long_name="Infrared brightness temperatures", units="K"),
    "ir_azimuth_angle": MetaData(
        long_name="Infrared sensor azimuth angle",
        standard_name="sensor_azimuth_angle",
        units="degree",
        comment="0=North, 90=East, 180=South, 270=West",
    ),
    "ir_elevation_angle": MetaData(
        long_name="Infrared sensor elevation angle",
        units="degree",
        comment="0=horizon, 90=zenith",
    ),
}


DEFINITIONS_1B21 = {
    "met_quality_flag": (
        "\n"
        "Bit 1: low_quality_air_temperature\n"
        "Bit 2: low_quality_relative_humidity\n"
        "Bit 3: low_quality_air_pressure\n"
        "Bit 4: low_quality_rainfall_rate\n"
        "Bit 5: low_quality_wind_direction\n"
        "Bit 6: low_quality_wind_speed"
    )
}

ATTRIBUTES_1B21 = {
    "air_temperature": MetaData(
        long_name="Air temperature",
        standard_name="air_temperature",
        units="K",
    ),
    "relative_humidity": MetaData(
        long_name="Relative humidity",
        standard_name="relative_humidity",
        units="1",
    ),
    "air_pressure": MetaData(
        long_name="Air pressure",
        standard_name="air_pressure",
        units="Pa",
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        standard_name="rainfall_rate",
        units="m s-1",
    ),
    "wind_direction": MetaData(
        long_name="Wind direction",
        standard_name="wind_from_direction",
        units="degree",
    ),
    "wind_speed": MetaData(
        long_name="Wind speed",
        standard_name="wind_speed",
        units="m s-1",
    ),
    "met_quality_flag": MetaData(
        long_name="Meteorological data quality flag",
        units="1",
        definition=DEFINITIONS_1B21["met_quality_flag"],
        comment="0=ok, 1=problem. Note: should also be set to 1\n"
        "if corresponding sensor not available",
    ),
}


FuncType: TypeAlias = Callable[[str], dict]
att_reader: dict[str, dict] = {
    "1B01": ATTRIBUTES_1B01,
    "1B11": ATTRIBUTES_1B11,
    "1B21": ATTRIBUTES_1B21,
}
