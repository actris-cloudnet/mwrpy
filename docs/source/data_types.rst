==========
Data Types
==========

The following data types (E-PROFILE naming convention) are available in MWRpy for Level 1 and Level 2 products:

Level 1
.......

- 1B01: MWR brightness temperatures from .BRT and .BLB/.BLS files + retrieved spectrum
- 1B11: IR brightness temperatures from .IRT files
- 1B21: Weather station data from .MET files
- 1C01: Combined data type with time corresponding to 1B01

Level 2
.......

- 2I01: Liquid water path (LWP)
- 2I02: Integrated water vapor (IWV)
- 2I06: Stability Indices
- 2P01: Temperature profiles from single-pointing observations
- 2P02: Temperature profiles from multiple-pointing observations
- 2P03: Absolute humidity profiles
- 2P04: Relative humidity profiles (derived from 2P01/2P02 + 2P03)
- 2P07: Potential temperature (derived from 2P01/2P02 + 2P03)
- 2P08: Equivalent potential temperature (derived from 2P01/2P02 + 2P03)
- single: Single pointing data product (including 2I01, 2I02, 2I06, 2P01, 2P03, and derived products)
- multi: Multiple pointing data product (including 2P02, and derived products)


The data types 1C01, single, and multi are also available in the Cloudnet format.
