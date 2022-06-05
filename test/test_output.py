"""
Test output of level 1 converter
"""
import glob
import os
import warnings

import numpy as np
import pint_xarray
import xarray as xr
from metpy.units import units

# Convert example level0 file with current script to level1
from pysonde.pysonde import main

pint_xarray.unit_Registry = units

l1_output_file_fmt = "pytest_l1-output_tmp_{direction}.nc"
l1_output_files = glob.glob(l1_output_file_fmt.format(direction="*"))

for file in l1_output_files:
    os.remove(file)

main(
    args={
        "inputfile": "examples/level0/BCO_20200126_224454.mwx",
        "config": "config/main.yaml",
        "output": l1_output_file_fmt,
        "verbose": "INFO",
    }
)

l2_output_file_fmt = "pytest_l2-output_tmp_{direction}.nc"

main(
    args={
        "inputfile": "./../pysonde/" + l1_output_file_fmt.format(direction="ascent"),
        "config": "config/main.yaml",
        "output": l2_output_file_fmt,
        "verbose": "INFO",
        "method": "bin",
    }
)

# import pdb;
# pdb.set_trace()


def test_file_consistency():
    """Test whether the old file (created with eurec4a_snd) agrees well with the new file"""
    ds_old = xr.open_dataset(
        "examples/level1/EUREC4A_BCO_Vaisala-RS_L1-ascent_20200126T2244_v3.0.0.nc"
    )
    ds_new = xr.open_dataset(l1_output_file_fmt.format(direction="ascent"))
    assert "ta" in ds_old.data_vars.keys(), "ta is missing"
    for var in ds_old.data_vars.keys():
        max_diff = (
            np.abs(
                ds_new.isel(sounding=0).reset_coords().squeeze()[var].pint.quantify()
                - ds_old.isel(sounding=0).reset_coords().squeeze()[var].pint.quantify()
            )
            .max()
            .pint.dequantify()
        )
        assert (
            max_diff <= 0.0001
        ), f"difference between old and new dataset is too large for {var}"


def test_sounding_id():
    ds_old = xr.open_dataset(
        "examples/level1/EUREC4A_BCO_Vaisala-RS_L1-ascent_20200126T2244_v3.0.0.nc"
    )
    ds_new = xr.open_dataset(l1_output_file_fmt.format(direction="ascent"))
    sounding_id_old_values = ds_old.sounding.values[0].split("__")
    sounding_id_new_values = ds_new.sounding.values[0].split("__")

    for v, (ov, nv) in enumerate(zip(sounding_id_old_values, sounding_id_new_values)):
        if v == 0 and not (ov == nv):
            warnings.warn(f"Platforms have changes from {ov} to {nv}")
        else:
            assert ov == nv, print(
                f"Parts of sounding IDs do not match (old: {ds_old.sounding}; new: {ds_new.sounding})"
            )
