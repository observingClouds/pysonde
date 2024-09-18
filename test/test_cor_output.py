"""
Test output of level 1 converter
"""

import glob
import os
from io import StringIO

import pint_xarray
from metpy.units import units
from omegaconf import OmegaConf

# Convert example level0 file with current script to level1
from pysonde.pysonde import main

pint_xarray.unit_Registry = units

cor_l1_output_file_fmt = "cor_pytest_l1-output_tmp_{direction}.nc"
cor_l1_output_files = glob.glob(cor_l1_output_file_fmt.format(direction="*"))
cor_l2_output_file_fmt = "cor_pytest_l2-output_tmp_{direction}.nc"

for file in cor_l1_output_files:
    os.remove(file)


def modify_main_yaml():
    """
    Change main.yaml to use level0_cor.yml for cor file conversion
    """
    main_yaml = OmegaConf.load("config/main.yaml")
    main_yaml["configs"]["level0"] = "config/level0_cor.yml"
    main_yaml_buffer = StringIO()
    OmegaConf.save(config=main_yaml, f=main_yaml_buffer)
    main_yaml_buffer.seek(0)
    return main_yaml_buffer


def test_cor_conversion_to_level1():
    main(
        args={
            "inputfile": "examples/level0/SA2024081600_1.cor",
            "config": modify_main_yaml(),
            "output": cor_l1_output_file_fmt,
            "verbose": "INFO",
        }
    )


def test_cor_conversion_to_level2():
    main(
        args={
            "inputfile": cor_l1_output_file_fmt.format(direction="ascent"),
            "config": modify_main_yaml(),
            "output": cor_l2_output_file_fmt,
            "verbose": "INFO",
            "method": "bin",
        }
    )
