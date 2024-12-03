File format description
=======================

All MWRpy files use ``NETCDF4_CLASSIC`` data model, i.e., ``HDF5`` file format.

**Dimensions**

.. list-table::
   :widths: 25
   :header-rows: 1

   * - Name
   * - time
   * - bnds
   * - frequency
   * - ir_wavelength
   * - receiver_nb
   * - t_amb_nb
   * - height


**Variables (common to all files)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - time
     - Time (UTC) of the measurement
     - time
     - seconds since 1970-01-01 00:00:00.000
     - float64
     - time
   * - time_bnds
     - Start and end time (UTC) of the measurements
     - time, bnds
     - seconds since 1970-01-01 00:00:00.000
     - int32
     -
   * - latitude
     - Latitude of measurement station
     - time
     - degree_north
     - float32
     - latitude
   * - longitude
     - Longitude of measurement station
     - time
     - degree_east
     - float32
     - longitude
   * - altitude
     - Altitude above mean sea level of measurement station
     - time
     - m
     - float32
     - altitude


MWR-Level 1 files
.................

1C01 file
~~~~~~~~~

The Level 1 default file type ``1C01`` contains all variables from the file types
``1B01``, ``1B11`` (if an infrared radiometer is available), and ``1B21`` (if a weather station is available).

**Variables (MWR_1B01 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - azimuth_angle
     - Azimuth angle
     - time
     - degree
     - float32
     - sensor_azimuth_angle
   * - elevation_angle
     - Elevation angle
     - time
     - degree
     - float32
     - sensor_elevation_angle
   * - tb
     - Microwave brightness temperature
     - time, frequency
     - K
     - float32
     - brightness_temperature
   * - frequency
     - Nominal centre frequency of microwave channels
     - frequency
     - GHz
     - float32
     -
   * - frequency_shift
     - Frequency shift of the microwave channels
     - frequency
     - GHz
     - float32
     -
   * - bandwidth
     - Bandwidth of microwave channels
     - frequency
     - GHz
     - float32
     -
   * - receiver
     - Corresponding microwave receiver for each channel
     - frequency
     - 1
     - int32
     -
   * - receiver_nb
     - Microwave receiver number
     - receiver_nb
     - 1
     - int32
     -
   * - n_sidebands
     - Number of sidebands
     - receiver_nb
     - 1
     - int32
     -
   * - sideband_IF_separation
     - Sideband IF separation
     - frequency
     - 1
     - float32
     -
   * - t_amb
     - Ambient target temperature
     - time, t_amb_nb
     - K
     - float32
     -
   * - t_rec
     - Receiver physical temperature
     - time, receiver_nb
     - K
     - float32
     -
   * - t_sta
     - Receiver temperature stability
     - time, receiver_nb
     - K
     - float32
     -
   * - pointing_flag
     - Pointing flag
     - time
     - 1
     - int32
     -
   * - quality_flag
     - Quality flag
     - time, frequency
     - 1
     - int32
     -
   * - quality_flag_status
     - Quality flag status
     - time, frequency
     - 1
     - int32
     -
   * - liquid_cloud_flag
     - Liquid cloud flag
     - time
     - 1
     - int32
     -
   * - liquid_cloud_flag_status
     - Liquid cloud flag status
     - time
     - 1
     - int32
     -

**Variables (MWR_1B11 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - ir_azimuth_angle
     - Infrared sensor azimuth angle
     - time
     - degree
     - float32
     - sensor_azimuth_angle
   * - ir_elevation_angle
     - Infrared sensor elevation angle
     - time
     - degree
     - float32
     - sensor_elevation_angle
   * - irt
     - Infrared brightness temperatures
     - time, ir_wavelength
     - K
     - float32
     -
   * - ir_wavelength
     - Wavelength of infrared channels
     - ir_wavelength
     - m
     - float32
     - sensor_band_central_radiation_wavelength
   * - ir_bandwidth
     - Bandwidth of infrared channels
     -
     - m
     - float32
     -
   * - ir_beamwidth
     - Beam width of the infrared radiometer
     -
     - degree
     - float32
     -

**Variables (MWR_1B21 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - air_temperature
     - Air temperature
     - time
     - K
     - float32
     - air_temperature
   * - relative_humidity
     - Relative humidity
     - time
     - 1
     - float32
     - relative_humidity
   * - air_pressure
     - Air pressure
     - time
     - Pa
     - float32
     - air_pressure
   * - rainfall_rate
     - Rainfall rate
     - time
     - m s-1
     - float32
     - rainfall_rate
   * - wind_speed
     - Wind speed
     - time
     - m s-1
     - float32
     - wind_speed
   * - wind_direction
     - Wind direction
     - time
     - degree
     - float32
     - wind_from_direction
   * - met_quality_flag
     - Meteorological data quality flag
     - time
     - 1
     - int32
     -

MWR-Level 2 files
...............

**Variables (common to all Level 2 files)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - azimuth_angle
     - Azimuth angle
     - time
     - degree
     - float32
     - sensor_azimuth_angle
   * - elevation_angle
     - Elevation angle
     - time
     - degree
     - float32
     - sensor_elevation_angle

Single pointing file
~~~~~~~~~~~~~~~~~~~~

The Level 2 default file type ``single`` contains all variables from the file types
``2I01``, ``2I02``, ``2I06``, ``2P01``, and ``2P03`` (if the respective retrieval coefficients are available).

**Variables (MWR_2I01 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - lwp
     - Liquid water path
     - time
     - kg m-2
     - float32
     - atmosphere_cloud_liquid_water_content
   * - lwp_offset
     - Subtracted offset correction of liquid water path
     - time
     - kg m-2
     - float32
     -
   * - lwp_quality_flag
     - Liquid water path quality flag
     - time
     - 1
     - int32
     -
   * - lwp_quality_flag_status
     - Liquid water path quality flag status
     - time
     - 1
     - int32
     -

**Variables (MWR_2I02 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - iwv
     - Integrated water vapour
     - time
     - kg m-2
     - float32
     - atmosphere_mass_content_of_water_vapor
   * - iwv_quality_flag
     - Integrated water vapour quality flag
     - time
     - 1
     - int32
     -
   * - iwv_quality_flag_status
     - Integrated water vapour quality flag status
     - time
     - 1
     - int32
     -

**Variables (MWR_2I06 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - lifted_index
     - Lifted index
     - time
     - K
     - float32
     -
   * - ko_index
     - KO index
     - time
     - K
     - float32
     -
   * - total_totals
     - Total totals index
     - time
     - K
     - float32
     -
   * - k_index
     - K index
     - time
     - K
     - float32
     -
   * - showalter_index
     - Showalter index
     - time
     - K
     - float32
     -
   * - cape
     - Convective available potential energy
     - time
     - J kg-1
     - float32
     -
   * - stability_quality_flag
     - Quality flag for stability products
     - time
     - 1
     - int32
     -
   * - stability_quality_flag_status
     - Quality flag status for stability products
     - time
     - 1
     - int32
     -

**Variable (common to all 2PXX files)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - height
     - Height above mean sea level
     - height
     - m
     - float32
     - height_above_mean_sea_level

**Variables (MWR_2P01 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - temperature
     - Temperature
     - time, height
     - K
     - float32
     - air_temperature
   * - temperature_quality_flag
     - Temperature quality flag
     - time
     - 1
     - int32
     -
   * - temperature_quality_flag_status
     - Temperature quality flag status
     - time
     - 1
     - int32
     -

**Variables (MWR_2P03 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - absolute_humidity
     - Absolute humidity
     - time, height
     - kg m-3
     - float32
     -
   * - absolute_humidity_quality_flag
     - Absolute humidity quality flag
     - time
     - 1
     - int32
     -
   * - absolute_humidity_quality_flag_status
     - Absolute humidity quality flag status
     - time
     - 1
     - int32
     -

Multiple pointing file
~~~~~~~~~~~~~~~~~~~~~~

The Level 2 default file type ``multi`` contains all variables from the file types
``2P02``, ``2P04``, ``2P07``, and ``2P08`` (if the respective retrieval coefficients are available).

**Variables (MWR_2P02 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - temperature
     - Temperature
     - time, height
     - K
     - float32
     - air_temperature
   * - temperature_quality_flag
     - Temperature quality flag
     - time
     - 1
     - int32
     -
   * - temperature_quality_flag_status
     - Temperature quality flag status
     - time
     - 1
     - int32
     -

**Variables (MWR_2P04 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - relative_humidity
     - Relative humidity
     - time, height
     - 1
     - float32
     - relative_humidity

**Variables (MWR_2P07 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - potential_temperature
     - Potential temperature
     - time, height
     - K
     - float32
     - air_potential_temperature

**Variables (MWR_2P08 specific)**

.. list-table::
   :widths: 25 50 25 25 25 25
   :header-rows: 1

   * - Name
     - Long name
     - Dimensions
     - Units
     - Data type
     - Standard name
   * - equivalent_potential_temperature
     - Equivalent potential temperature
     - time, height
     - K
     - float32
     - air_equivalent_potential_temperature
