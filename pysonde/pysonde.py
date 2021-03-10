#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert sounding files from different sources
to netCDF
"""
import argparse
import logging
import sys

import tqdm
from omegaconf import OmegaConf

from . import _helpers as h


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--inputfile",
        metavar="INPUT_FILE",
        help="Single sonde file or file format\n" "including wildcards",
        default=None,
        required=False,
        nargs="+",
        type=h.unixpath,
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="/some/example/path/",
        help="Output folder for converted files (netCDF). You can\n"
        " although not recommended also define an output file\n"
        "(format). However, please share only those with the\n"
        " the default filename.\n"
        " The following formats can be used:\n"
        "\t {platform}\t platform name\n"
        "\t {location}\t platform location\n"
        "\t {direction}\t sounding direction\n"
        "\t\t date format with\n"
        "\t\t %%Y%%m%%d %%H%%M and so on\n"
        "\t\t and others to format the output folder dynamically.",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="MAIN_CONFIG.YML",
        help="Main config file with references\n" "to specific config files",
        default="../config/main.yaml",
        required=False,
        type=h.unixpath,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        metavar="DEBUG",
        help="Set the level of verbosity [DEBUG, INFO," " WARNING, ERROR]",
        required=False,
        default="INFO",
    )

    parsed_args = vars(parser.parse_args())

    if parsed_args["inputfile"] is None:
        parser.error(
            "--inputfile must be defined. For several files"
            "enter the inputfile format with wildcards."
        )

    return parsed_args


def find_files(arg_input):
    """
    Find files to convert
    """
    if isinstance(arg_input, list) and len(arg_input) > 1:
        filelist = arg_input
    elif isinstance(arg_input, list) and len(arg_input) == 1:
        filelist = h.expand_pathglobs(arg_input[0])
    elif isinstance(arg_input, str):
        filelist = h.expand_pathglobs(arg_input)
    else:
        raise ValueError
    return sorted(filelist)


def load_reader(filename):
    """
    Infer appropriate reader from filename
    """
    ending = filename.suffix
    if ending == ".mwx":
        from .readers.readers import MW41

        reader = MW41
    else:
        raise h.ReaderNotImplemented(f"Reader for filetype {ending} not implemented")
    return reader


def main(args=None):
    if args is None:
        args = {}
    try:
        args = get_args()
    except ValueError:
        sys.exit()

    h.setup_logging(args["verbose"])

    # Combine all configurations
    main_cfg = OmegaConf.load(args["config"])
    cfg = h.combine_configs(main_cfg.configs)

    cfg = h.replace_placeholders_cfg(cfg)

    input_files = find_files(args["inputfile"])
    logging.info("Files to process {}".format([file.name for file in input_files]))

    logging.debug("Load reader. All files need to be of same type!")
    # Load correct reader class
    reader_class = load_reader(input_files[0])
    # Configure reader according to config file
    reader = reader_class(cfg)

    for ifile, file in enumerate(tqdm.tqdm(input_files)):
        logging.debug("Reading file number {}".format(ifile))
        sounding = reader.read(file)
        # Split sounding into ascending and descending branch
        sounding_asc, sounding_dsc = sounding.split_by_direction()
        for snd in [sounding_asc, sounding_dsc]:
            if len(snd.profile) < 2:
                logging.warning(
                    "Sounding ({}) does not contain data. "
                    "Skip sounding-direction of {}".format(
                        snd.meta_data["sounding_direction"], file
                    )
                )
                continue
            snd.calculate_additional_variables(cfg)
            snd.convert_sounding_df2ds()
            snd.create_dataset(cfg)
            snd.export(args["output"], cfg)


if __name__ == "__main__":
    main()
