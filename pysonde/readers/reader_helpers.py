"""Helper functions for the different readers
"""
import glob
import os
import shutil
import tempfile
import warnings
import zipfile
from typing import Optional, Type, Union
from xml.dom import minidom

import numpy as np
import pandas as pd
import xarray as xr


class VariableNotFoundInSounding(Warning):
    pass


class SondeTypeNotIdentifiable(Warning):
    pass


def custom_formatwarning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    # ignore everything except the message
    return f"{message}\n"


warnings.formatwarning = custom_formatwarning


# Vaisala MW41
def getTmpDir():
    """
    Creates a temporary folder at the systems default location for temporary files.

    Returns:
        Sets Class variables:
        - self.tmpdir_obj: tempfile.TemporaryDirectory
        - self.tmpdir: string containing the path to the folder
    """
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_obj.name
    return tmpdir, tmpdir_obj


def decompress(file, tmp_folder):
    """
    Decompress file to temporary folder
    """
    with zipfile.ZipFile(file) as z:
        z.extractall(tmp_folder)
    decompressed_files = sorted(glob.glob(os.path.join(f"{tmp_folder}", "*")))

    return decompressed_files


def compress(folder, compressed_file):
    """
    Compress folder to compressed file
    """
    archive = shutil.make_archive(compressed_file, "zip", folder)
    os.rename(archive, compressed_file)
    return


def open_mwx(mwx_file):
    """
    Open Vaisala MWX41 archive file (.mwx)

    Input
    -----
    mwx_file : str
        Vaisala MW41 archive file

    Returns
    -------
    decompressed_files : list
        List of temporarily decompressed .xml files
        within the archive file
    """
    tmpdir, tmpdir_obj = getTmpDir()
    decompressed_files = np.array(decompress(mwx_file, os.path.join(tmpdir, "")))
    return decompressed_files


class MWX(object):
    """
    Open Vaisala MWX41 archive file (.mwx)

    Input
    -----
    mwx_file : str
        Vaisala MW41 archive file

    Returns
    -------
    decompressed_files : list
        List of temporarily decompressed .xml files
        within the archive file
    """

    def __init__(self, mwx_file):
        self.tmpdir, self.tmpdir_obj = getTmpDir()
        self.decompressed_files = np.array(
            decompress(mwx_file, os.path.join(self.tmpdir, ""))
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.tmpdir_obj.cleanup()

    def get_decompressed_files(self):
        return self.decompressed_files


def check_availability(decomp_files, file, return_name=False):
    """
    Check whether xml file exist in decompressed
    file list

    Returns
    -------
    avail : bool
        Availability of file
    filename : str (optional)
        Full filename of requested file
    """
    basenames = [os.path.basename(decomp_file) for decomp_file in decomp_files]

    # Availability
    availability_mask = np.in1d(basenames, file)

    if np.sum(availability_mask) > 0:
        avail = True
    else:
        avail = False

    if return_name:
        if avail:
            idx = np.where(availability_mask)[0][0]
            fullname = decomp_files[idx]
        else:
            fullname = None
        return avail, fullname
    else:
        return avail


def read_xml(filename, return_handle=False):
    xmldoc = minidom.parse(filename)
    itemlist = xmldoc.getElementsByTagName("Row")
    if return_handle:
        return itemlist, xmldoc
    else:
        return itemlist


def get_sounding_profile(file, keys):
    """
    Get sounding profile from provided xml file

    Input
    -----
    file : str
        XML file containing sounding data e.g.
        SynchronizedSoundingData.xml
    keys : list
        list of variables to look for

    Returns
    -------
    pd_snd : pandas.DataFrame
        sounding profile
    """
    itemlist = read_xml(file)
    sounding_dict = {}
    try:
        for i, item in enumerate(itemlist):
            level_dict = {}
            for var in keys:
                level_dict[var] = item.attributes[var].value
            sounding_dict[i] = level_dict
    except KeyError:
        warnings.warn("Key {} not found.".format(var), VariableNotFoundInSounding)
    pd_snd = pd.DataFrame.from_dict(sounding_dict, orient="index")
    types = {c: float for c in pd_snd.columns}
    types["SoundingIdPk"] = str
    types["DataSrvTime"] = str
    types["PtuStatus"] = int
    types["WindInterpolated"] = bool
    types["Dropping"] = int
    pd_snd = pd_snd.astype(types)

    # Set missing values to NaN
    pd_snd = pd_snd.replace(-32768, np.nan)
    return pd_snd


def get_sounding_metadata(file, keys):
    itemlist = read_xml(file, False)
    sounding_meta_dict = {}

    for i, item in enumerate(itemlist):
        assert (
            i == 0
        ), "further entries were found, meaning soundings meta data could be mixed up"
        for var in keys:
            try:
                sounding_meta_dict[var] = item.attributes[var].value
            except KeyError:
                warnings.warn(
                    "Attribute {} could not found and is assumed to be RS41-SGP".format(
                        var
                    ),
                    SondeTypeNotIdentifiable,
                )
                sounding_meta_dict[var] = "RS41-SGP"
    return sounding_meta_dict


def rename_variables(sounding, variable_dict):
    """Rename variables in sounding
    according to key, value pairs in
    variable dict
    """
    if isinstance(sounding, xr.core.dataset.Dataset):
        vars = list(sounding.data_vars.keys())
        vars.extend(list(sounding.coords.keys()))
    else:
        vars = sounding.columns
    rename_dict = {}
    for var in vars:
        if var in variable_dict.keys():
            rename_dict[var] = variable_dict[var]

    if isinstance(sounding, xr.core.dataset.Dataset):
        sounding = sounding.rename(rename_dict)
    else:
        sounding = sounding.rename(columns=rename_dict)

    return sounding


def rename_metadata(meta_dict, variable_dict):
    """Rename variables of sounding metadata
    according to key, value pairs in
    variable dict
    """
    updated_dict = {}
    for var in meta_dict.keys():
        if var in variable_dict.keys():
            updated_dict[variable_dict[var]] = meta_dict[var]
        else:
            updated_dict[var] = meta_dict[var]
    return updated_dict
