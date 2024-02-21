================
MWRpy processing
================

In this tutorial `MWRpy <https://github.com/actris-cloudnet/mwrpy/>`_ products are generated from raw data, including
quality control and visualization. This example utilizes files taken from the ACTRIS site
`Hyytiala <https://cloudnet.fmi.fi/site/hyytiala>`_:

- RPG-HATPRO microwave radiometer:
    - Brightness temperatures: `230406.BRT <https://github.com/actris-cloudnet/mwrpy/blob/main/tests/data/hyytiala/230406.BRT>`_
    - Housekeeping data: `230406.HKD <https://github.com/actris-cloudnet/mwrpy/blob/main/tests/data/hyytiala/230406.HKD>`_
    - Elevation scans: `230406.BLB <https://github.com/actris-cloudnet/mwrpy/blob/main/tests/data/hyytiala/230406.BLB>`_
    - Weather station: `230406.MET <https://github.com/actris-cloudnet/mwrpy/blob/main/tests/data/hyytiala/230406.MET>`_
    - Infrared radiometer: `230406.IRT <https://github.com/actris-cloudnet/mwrpy/blob/main/tests/data/hyytiala/230406.IRT>`_

.. note::

    .BRT and .HKD files are mandatory in MWRpy for processing

Level 1c
~~~~~~~~~

Next we convert RPG-HATPRO microwave radiometer (MWR) binary files, including brightness temperature and housekeeping
data (\*.BRT, \*.HKD), into a Level 1c netCDF file. Data from optional elevation scans (\*.BLB, \*.BLS), weather
station (\*.MET) and infrared radiometer (\*.IRT) are combined in this process and quality flags are derived. It is
expected to have corresponding RPG retrieval coefficient files (``.RET``) in ``/data/hatpro-mwrpy-coeffs/``.

.. code-block:: python

    import glob
    from mwrpy.level1.write_lev1_nc import lev1_to_nc

    coeff_files = glob.glob(f"/data/hatpro-mwrpy-coeffs/*.ret")
    site_meta = {'name': 'Hyytiala', "coefficientFiles": coeff_files}
    coeff_files = site_meta.get("coefficientFiles", None)
    hatpro_raw = lev1_to_nc(
        "1C01",
        ".",
        output_file="mwr_1c",
        coeff_files=coeff_files,
        instrument_config=site_meta,
    )

The data format of the generated ``mwr_1c.nc`` file, including metadata information and variable names, is
compliant with the data structure and naming convention developed in the EUMETNET Profiling Programme
`E-PROFILE <https://www.eumetnet.eu/>`_.

Variables such as brightness temperature can be plotted from the newly generated file.

.. code-block:: python

    generate_figure('mwr_1c.nc', ['tb'])

.. figure:: _static/20230406_hyytiala_tb.png

Level 2 Single Pointing
~~~~~~~~~~~~~~~~~~~~~~~

Based on the Level 1c netCDF file ``mwr_1c.nc``, MWR single pointing data are extracted
and product specific retrieval coefficients are applied to generate the Level 2 single pointing product:

.. code-block:: python

    from mwrpy.level2.lev2_collocated import generate_lev2_single
    hatpro_prod = generate_lev2_single("hyytiala", "mwr_1c.nc", "mwr-single.nc")

Variables such as integrated water vapor
(`IWV <https://vocabulary.actris.nilu.no/skosmos/actris_vocab/en/page/watervapourtotalcolumncontent>`_)
can be plotted from the newly generated file.

.. code-block:: python

    generate_figure('mwr-single.nc', ['iwv'])

.. figure:: _static/20230406_hyytiala_iwv.png

Level 2 Multiple Pointing
~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the Level 1c file, MWR multiple pointing data (elevation scans) are extracted
and product specific retrieval coefficients are applied to generate the Level 2 multiple pointing product:

.. code-block:: python

    from mwrpy.level2.lev2_collocated import generate_lev2_multi
    hatpro_prod = generate_lev2_multi("hyytiala", "mwr_1c.nc", "mwr-multi.nc")

Variables such as temperature profiles can be plotted from the newly generated file.

.. code-block:: python

    generate_figure('mwr-multi.nc', ['temperature'])

.. figure:: _static/20230406_hyytiala_temperature.png
