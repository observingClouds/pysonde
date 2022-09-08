# pySonde: converting radiosonde files to netcdf

pySonde converts Vaisala's radiosonde files (mwx) to netCDF4 and interpolates them if needed to a common height grid for easier processing.

## Setup
```
pip install pysonde
```

For development
```sh
# Install dependencies
pipenv install --dev

# Setup pre-commit and pre-push hooks
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```

## Usage

A few example files are automatically installed and can be used to test if the installation was successful

Unix:
```sh
sounding_converter -i examples/level0/BCO_20200126_224454.mwx -o "test_{direction}.nc" -c config/main.yaml
```

Windows:
```sh
sounding_converter.exe -i examples/level0/BCO_20200126_224454.mwx -o "test_{direction}.nc" -c config/main.yaml
```

The configuration of attributes, variable names and units of the input and output is done via yaml files in the `config` folder.

To post-process radiosoundings with pysonde and track the processing steps, a new repository should be created that only contains the `config` folder and its scripts. An additional bash script with the `sounding_converter` calls tracks the processing steps. The version used of pysonde is automatically inserted into the output files.

The [repository containing the processing setup for the circBrazil campaign](https://github.com/observingClouds/soundings_circbrazil) can serve as a template.
