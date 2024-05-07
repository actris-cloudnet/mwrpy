---
title: "MWRpy: A Python package for processing microwave radiometer data"
tags:
  - Python
  - meteorology
  - remote sensing
  - microwave radiometer
authors:
  - name: Tobias Marke
    corresponding: true
    orcid: 0000-0001-7804-9056
    affiliation: 1
  - name: Ulrich Löhnert
    orcid: 0000-0002-9023-0269
    affiliation: 1
  - name: Simo Tukiainen
    orcid: 0000-0002-0651-4622
    affiliation: 2
  - name: Tuomas Siipola
    affiliation: 2
    orcid: 0009-0004-7757-0893
  - name: Bernhard Pospichal
    orcid: 0000-0001-9517-8300
    affiliation: 1

affiliations:
  - name: Institute for Geophysics and Meteorology, University of Cologne, Germany
    index: 1
  - name: Finnish Meteorological Institute, Helsinki, Finland
    index: 2
date: 21 February 2024
bibliography: paper.bib
---

# Summary

Ground-based passive microwave radiometers (MWRs) are deployed to obtain information on the vertical structure of
temperature and water vapor mostly in the lower troposphere. In addition, they are used to derive the total column
integrated liquid water content of the atmosphere, referred to as liquid water path (LWP). MWRs measure radiances,
given as brightness temperatures ($T_B$), typically in two frequency ranges along absorption features of water vapor
and oxygen, as well as in window regions where the observations are sensitive to liquid water clouds. Profiles of
temperature and humidity are retrieved together with the vertically integrated water vapor content (IWV) and LWP
(e.g. @Crewell2003, @Löhnert2012). A specific elevation scanning configuration allows for an improved resolution for
temperature profiles in the atmospheric boundary-layer [@Crewell2007]. The instruments can be operated continuously
and provide temporally highly resolved observations of up to 1$s$, which make them a valuable tool for improving
numerical weather forecast and climate models by studying the atmospheric water cycle, including cloud dynamics
[@Westwater2004].

One widely used application exploiting MWR data is the synergistic algorithm Cloudnet [@IllingworthEt2007], which
classifies hydrometeors in the atmosphere by combining several ground-based remote sensing instruments. As part of
the European Aerosol, Clouds and Trace Gases Research Infrastructure (ACTRIS, @Laj2024), the Centre for
Cloud Remote Sensing (CCRES) is aiming to provide continuous and long-term data of cloud properties and the
thermodynamic state of the atmosphere, with Cloudnet being one of the key tools. For atmospheric observatories, MWRs
are therefore mandatory to qualify as an ACTRIS-CCRES compatible station. The ACTRIS Central Facility responsible
for MWRs in the network is hosted within ACTRIS Germany (ACTRIS-D).

The European cloud remote sensing network will encompass around 30 stations, including mobile platforms, and covering
different climatological zones. This network configuration enables investigations of similarities of atmospheric
processes and long-term trends between those sites. Some of the participating stations have been operational already
for more than a decade and Cloudnet products were derived based on their individual setup and processing
algorithms. To ensure that the generated data sets are comparable, station operators are required to follow the
ACTRIS-CCRES standard operating procedures and send raw data files to the central cloud remote sensing data center unit
(CLU, http://cloudnet.fmi.fi). CLU provides data storage and provision, but also the centralized processing,
including visualization, in order to harmonize the data streams.

# Statement of need

[MWRpy](https://actris-cloudnet.github.io/mwrpy/index.html#) addresses the needs of a centralized processing,
quality control of MWR raw data, and deriving standardized output of meteorological variables. The Python code is an
advancement of the IDL based processing software mwr_pro [@mwr_pro] and is able to handle raw data from HATPRO
manufactured by Radiometer Physics GmbH (RPG, https://www.radiometer-physics.de/), which is so far the only
instrument type in the network. The output format, including metadata information, variable names, and file
naming is designed to be compliant with the data structure and naming convention developed together with the
EUMETNET Profiling Programme E-PROFILE [@Rüfenacht2021]. In this way,
[MWRpy](https://actris-cloudnet.github.io/mwrpy/index.html#) improves data compatibility and fosters cross network
collaborations. The processing chain is replacing the mode of operation in Cloudnet, which previously relied on
pre-processed and non-harmonized MWR data, and therefore contributes to more ACTRIS data consistency. Statistical
analysis of these consistent long-term data sets is expected to be beneficial not only for atmospheric studies, but
also for improving knowledge on instrument operation and maintenance by monitoring key parameters from the
instrument and mandatory regular absolute calibrations (approximately every 6 months). Future developments include
the support of further instrument types, if present in the network. Furthermore, the flexible design of the code
enables updating the retrievals of meteorological variables, which will be derived from a common approach.

# Code design

[MWRpy](https://actris-cloudnet.github.io/mwrpy/index.html#) is designed to be used as a stand-alone software since
it covers the full processing and visualization chain from raw data to higher level products, but it is also
embedded in the Python implementation of the Cloudnet processing scheme CloudnetPy [@Tukiainen2020]. At first, data
quality control is performed on the mandatory data fields of measured $T_B$ and instrument specific
housekeeping data to generate quality flags. In a next step auxiliary data (e.g. from a weather station) are
combined to produce daily netCDF files. Subsequently advanced meteorological variables are derived by applying
retrieval coefficients and stored as separate daily files for variables originating from elevation scans (e.g.
temperature profiles) and all remaining measuring modes (including vertical stare for e.g. LWP). Within the Cloudnet
processing framework the output of [MWRpy](https://actris-cloudnet.github.io/mwrpy/index.html#) is then harmonized
and utilized by CloudnetPy, together with data streams from other ACTRIS-CCRES instruments, like cloud radar, to
derive synergy products. All files, including calibration and retrieval information, and corresponding
visualizations are stored in the Cloudnet data portal and accessible through an API.

# Acknowledgements

This work is funded by the Federal Ministry of Education and Research (BMBF) under the FONA Strategy “Research for
Sustainability” and part of the implementation of ACTRIS Germany (ACTRIS-D) under the research grant no. 01LK2002F.
The operation of the Central Facilities is supported by the Federal Ministry for the Environment, Nature
Conservation, Nuclear Safety and Consumer Protection (BMUV). The implementation and operation of ACTRIS-D is
co-funded by 11 German research performing organizations.

# References
