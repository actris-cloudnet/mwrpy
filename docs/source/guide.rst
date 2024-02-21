Developer's Guide
=================

MWRpy is hosted by the Institute for Geophysics and Meteorology at the University of Cologne and
is used to process microwave radiometer data in the
ACTRIS research infrastructure. We are happy to welcome the cloud remote sensing
community to provide improvements in the methods and their implementations,
writing tests and fixing bugs.

How to contribute
-----------------

Instructions can be found from
`MWRpy's Github page <https://github.com/actris-cloudnet/mwrpy/blob/main/CONTRIBUTING.md>`_.

Testing
-------

To run the MWRpy test suite, first
clone the whole repository from `GitHub
<https://github.com/actris-cloudnet/mwrpy>`_:

.. code-block:: console

	$ git clone https://github.com/actris-cloudnet/mwrpy

Testing environment
...................

Create a virtual environment and install:

.. code-block:: console

    $ cd mwrpy
    $ python3 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip3 install --upgrade pip
    (venv) $ pip3 install .[test,dev]

Example test
............

.. code-block:: console

    (venv) $ python3 tests/test_file_reading.py
