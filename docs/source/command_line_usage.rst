==================
Command line usage
==================

With the instrument type configuration (``mwrpy/site_config/{instrument_type}.yaml``) and retrieval files in
``mwrpy/site_config/{site}/coefficients/``, MWRpy can also be run using the command line tool `mwrpy/cli.py`:

.. code-block::

    mwrpy/cli.py [-h] -s SITE [-d YYYY-MM-DD] [--start YYYY-MM-DD]
                           [--stop YYYY-MM-DD] [-p ...] [{process,plot}]

.. list-table:: Arguments
   :widths: 10 20 20 50
   :header-rows: 1

   * - Short
     - Long
     - Default
     - Description
   * - `-h`
     - `--help`
     -
     - Show help and exit.
   * - `-s`
     - `--site`
     -
     - Site to process data from, e.g, `hyytiala`. Required.
   * - `-d`
     - `--date`
     -
     - Single date to be processed. Alternatively, `--start` and `--stop` can be defined.
   * -
     - `--start`
     - `current day - 1`
     - Starting date.
   * -
     - `--stop`
     - `current day`
     - Stopping date.
   * - `-p`
     - `--products`
     - 1C01, single, multi
     - Processed products, e.g, `1C01, 2I02, 2P03, single`, see Data Types.
   * - `-f`
     - `--format`
     - cloudnet
     - Data format to be used (`cloudnet`, `e-profile`).
   * - `-i`
     - `--instrument`
     - hatpro
     - Instrument type to be processed (`hatpro`, `lhatpro`, `lhumpro_u90`).

These commands are available to select the processing mode:

.. list-table:: Commands
   :widths: 20 30
   :header-rows: 1

   * - Command
     - Description
   * - `process`
     - Process data and generate plots (default).
   * - `plot`
     - Only generate plots.
   * - `no-plot`
     - Only generate products.
   * - `reprocess`
     - Like `process`, but skips days when data processing fails.

Example usage
-------------
To process and plot Level 1 & 2 data (1C01, single, multi) for the site `Hyytiala` (HATPRO instrument) for April 6,
2023, in the Cloudnet format, run:

.. code-block::

    python mwrpy/cli.py -s hyytiala -d 2023-04-06

Run the following command for the E-Profile format and no plots:

.. code-block::

    python mwrpy/cli.py -s hyytiala -d 2023-04-06 -f e-profile no-plot
