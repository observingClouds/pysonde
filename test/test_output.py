import glob
import os

import numpy as np
import xarray as xr

# Convert example level0 file with current script to level1
from pysonde.pysonde import main

output_file_fmt = "pytest_output_tmp_{direction}.nc"
output_files = glob.glob(output_file_fmt.format(direction="*"))
for file in output_files:
    os.remove(file)
main(
    args={
        "inputfile": "examples/level0/BCO_20200126_224454.mwx",
        "config": "config/main.yaml",
        "output": output_file_fmt,
        "verbose": "INFO",
    }
)


def test_file_consistency():
    """Test whether the old file (created with eurec4a_snd) agrees well with the new file"""
    ds_old = xr.open_dataset(
        "examples/level1/EUREC4A_BCO_Vaisala-RS_L1-ascent_20200126T2244_v3.0.0.nc"
    )
    ds_new = xr.open_dataset(output_file_fmt.format(direction="ascent"))
    assert "ta" in ds_old.data_vars.keys(), "ta is missing"
    for var in ds_old.data_vars.keys():
        max_diff = np.abs(
            ds_new.isel(sounding=0)[var] - ds_old.isel(sounding=0)[var]
        ).max()
        assert (
            max_diff <= 0.0001
        ), f"difference between old and new dataset is too large for {var}"


def test_sounding_id():
    ds_old = xr.open_dataset(
        "examples/level1/EUREC4A_BCO_Vaisala-RS_L1-ascent_20200126T2244_v3.0.0.nc"
    )
    ds_new = xr.open_dataset(output_file_fmt.format(direction="ascent"))
    assert ds_old.sounding == ds_new.sounding, "Sounding IDs do not agree"
