Configuration
=============

The configuration of the converter is done via several yaml files that can be found in the config folder. It is recommended to copy these files and adapt them to ones needs. With the `-c / --config` command line arguement to `sounding-converter` the updated configuration can be used. Note, that the complete path to the `main.yaml` needs to be given. Further configurations are referenced within the `main.yaml`.

General configuration
---------------------
The general settings that are valid across processing levels are set in `main.yaml`. These include e.g. the platform name and the name of the campaign the soundings are associated with.

.. code-block:: yaml

   campaign: EUREC4A
   campaign_id: EUREC4A
   platform: BCO
   configs:
    main: config/main.yaml  # name of this file
    level0: config/level0_mwx.yml
    level1: config/level1.yaml

It also includes references to the other configuration files that are more specific.

Level0 Input
------------
The raw-data file (mwx) is described in `level0_mwx.yaml`. In general, there should be no need to make changes to this configuration, as long as the output (keys, units) in the MWX files does not change.

Level1 Output
-------------
The first processing step, the conversion of the mwx files, is configured in `level1.yaml`. This file consists of three sections, the *global attributes*, the *coordinates* and the *variables*.

Global attributes
^^^^^^^^^^^^^^^^^
The global attribute section contains all the attributes that shall appear in the Level 1 netCDF file. This section can be shortened or extended by simply adding further key, value pairs.

.. code-block:: yaml

  global_attrs:
    title: Level 1 sounding data
    featureType: "trajectory"
    Conventions: "CF-1.7"

Special variables (`${main}`, `${meta_level0}`) or actually pointers can be used to reference values in other config files. This way the attribute *title* can for example by dynamically set with the settings in the main configuration file:

.. code-block:: yaml

  global_attrs:
    title: ${main.campaign} level 1 sounding data

Further some attributes have specific placeholders, that are filled at run-time. One example is the *history*-attribute:

.. code-block:: yaml

    history: "created with {package} ({version}) on {date}"

Coordinates
^^^^^^^^^^^
The coordinate section consists of at least the coordinate names and the dimension key that includes the dimension of the dataset. Since this dimension is mostly not known before runtime, references are typically made to a runtime dictionary.

.. code-block:: yaml

  coordinates:
    sounding:
        dimension: ${runtime.sounding_dim}
    level:
        dimension: ${runtime.level_dim}

Variables
^^^^^^^^^
This section contains the variables, that shall be written in the level 1 netcdf file. Whole section can be deleted or added. Of importance is the key *internal_varname* that connects the output variable with the internally used variable name.
The variable attributes like standard_name, long_name or any other user defined attributes are set under *attrs*. Of special importance is the *units* attribute as it determines the unit of the output. Due to the unit-awareness of `pysonde` the input data is automatically converted to the given unit. These units therefore need to be valid Pint units.

The datatype of the output can be given in the encodings section.

The coordinates that describe the variable are given as a list below *coordinates*

.. code-block:: yaml

      ta:
        attrs:
            standard_name: "air_temperature"
            long_name: "dry bulb temperature"
            units: "K"
            coordinates: "launch_time flight_time lon lat p"
        encodings:
            dtype: "float32"
        coordinates:
            - sounding
            - level
        internal_varname : "temperature"


