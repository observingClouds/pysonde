import inspect
import logging
import platform
import subprocess as sp
import re
import json
import time
from pathlib import Path, PureWindowsPath

from omegaconf import OmegaConf


class ReaderNotImplemented(Exception):
    pass


class RegexDict(dict):
    """
    Dictionary with capability of taking regular expressions
    """
    def get_matching(self, event):
        return (self[key] for key in self if re.match(key, event))

    def get_matching_combined(self, event):
        """
        Find matching keys and return combined dictionary
        >>> d = {'EUREC4A_*':{'a':0}, 'EUREC4A_BCO':{'p':1}}
        >>> rd = RegexDict(d)
        >>> rd.get_matching_combined('EUREC4A_BCO')
        {'a': 0, 'p': 1}
        """
        matching = self.get_matching(event)
        dall = {}
        for d in matching:
            dall.update(d)
        return dall


def get_version():
    logging.debug("Gathering version information")
    version = "--"
    try:
        import pysonde

        version = pysonde.__version__
    except (ModuleNotFoundError, AttributeError):
        logging.debug("No pysonde package version found")

    try:
        version = (
            sp.check_output(
                ["git", "describe", "--always", "--dirty"], stderr=sp.STDOUT
            )
            .strip()
            .decode()
        )
    except (sp.CalledProcessError, FileNotFoundError):
        logging.debug("No git-version could be found.")

    return version


# def get_inputfile():
    
#     return inputfile


def replace_placeholders_cfg(cfg, subset="level2"):
    """
    Replace placeholders in config that only exist during
    runtime e.g. time, version, ...
    """
    if "history" in cfg[subset].global_attrs.keys():
        version = get_version()
        cfg[subset].global_attrs["history"] = (
            cfg[subset]
            .global_attrs["history"]
            .format(
                version=version, package="pysonde", date=str(time.ctime(time.time()))
            )
        )
    if "version" in cfg[subset].global_attrs.keys():
        version = get_version()
        cfg[subset].global_attrs["version"] = (
            cfg[subset].global_attrs["version"].format(version=version)
        )

    return cfg


def unixpath(path_in):
    """
    Convert windows path to unix path syntax
    depending on the used OS
    """
    if platform.system() == "Windows":
        path_out = Path(PureWindowsPath(path_in))
    else:
        path_out = Path(path_in)
    return path_out


def find_files(arg_input):
    """
    Find files to convert
    """
    if isinstance(arg_input, list) and len(arg_input) > 1:
        filelist = arg_input
    elif isinstance(arg_input, list) and len(arg_input) == 1:
        filelist = expand_pathglobs(arg_input[0])
    elif isinstance(arg_input, str):
        filelist = expand_pathglobs(arg_input)
    else:
        raise ValueError
    return sorted(filelist)


def expand_pathglobs(pathparts, basepaths=None):
    """
    from https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
    Logic:
     0. Argue with a Path(str).parts and optional ['/start','/dirs'].
     1. for each basepath, expand out pathparts[0] into "expandedpaths"
     2. If there are no more pathparts, expandedpaths is the result.
     3. Otherwise, recurse with expandedpaths and the remaining pathparts.
     eg: expand_pathglobs('/tmp/a*/b*')
       --> /tmp/a1/b1
       --> /tmp/a2/b2
    """
    if isinstance(pathparts, str) or isinstance(pathparts, Path):
        pathparts = Path(pathparts).parts

    if basepaths is None:
        return expand_pathglobs(pathparts[1:], [Path(pathparts[0])])
    else:
        assert pathparts[0] != "/"

    expandedpaths = []
    for p in basepaths:
        assert isinstance(p, Path)
        globs = p.glob(pathparts[0])
        for g in globs:
            expandedpaths.append(g)

    if len(pathparts) > 1:
        return expand_pathglobs(pathparts[1:], expandedpaths)

    return expandedpaths


def setup_logging(verbose):
    assert verbose in ["DEBUG", "INFO", "WARNING", "ERROR"]
    # Get filename of calling script
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__

    logging.basicConfig(
        level=logging.getLevelName(verbose),
        format="%(levelname)s - %(name)s - %(funcName)s - %(message)s",
        handlers=[
            logging.FileHandler("{}.log".format(filename)),
            logging.StreamHandler(),
        ],
    )


def combine_configs(config_dict):
    """
    Combine Omega configs given as dictionary
    """
    return OmegaConf.merge(
        {config: OmegaConf.load(path) for config, path in config_dict.items()}
    )


def remove_nontype_keys(dict, allowed_type=type("str")):
    """
    Remove keys from dictionary that have another type
    than the once allowed.
    """
    return {k: v for (k, v) in dict.items() if isinstance(v, allowed_type)}


def remove_missing_cfg(cfg):
    """
    Remove config keys that are missing
    """
    return_cfg = {}
    for k in cfg.keys():
        if OmegaConf.is_missing(cfg, k):
            logging.warning(f"key {k} is missing and skipped")
            pass
        else:
            return_cfg[k] = cfg[k]
    return OmegaConf.create(return_cfg)


def replace_global_attributes(ds, cfg, subset='level2'):
    logging.debug("Replace global attributes that change in comparison to the level1 data")
    
    cfg = replace_placeholders_cfg(cfg)
    for k in cfg[subset].global_attrs.keys():
        ds.attrs[k] = cfg[subset].global_attrs[k]
        
    return ds


def set_global_attributes(ds, cfg):
    logging.debug("Add global attributes")
    
    _cfg = remove_missing_cfg(cfg)
    ds.attrs = _cfg
                
    return ds


def set_additional_var_attributes(ds, meta_data_dict, variables):
    """
    Set further descriptive variable
    attributes and encoding.
    """
    for var_in, var_out in variables:
        try:
            meta_data_var = meta_data_dict[var_in]['attrs']
            for key, value in meta_data_var.items():
                if key not in ['_FillValue', 'dtype'] and not ('time' in var_out):
                    ds[var_out].attrs[key] = value
                elif (key not in ['_FillValue', 'dtype', 'units']) and ('time' in var_out):
                    ds[var_out].attrs[key] = value
                elif (key == '_FillValue') and (value == False):
                    ds[var_out].attrs[key] = value
                else:
                    ds[var_out].encoding[key] = value
                    
        except KeyError:
            continue
     
    return ds


def compress_dataset(ds):
    """
    Apply internal netCDF4 compression
    """
    for var in ds.data_vars:
        ds[var].encoding['zlib'] = True
    return ds


def write_dataset(ds, filename):
    ds = compress_dataset(ds)
    # check correct units here, compare with level 1 how this is done there (link Hauke sent on Mattermost)
    ds = ds.pint.dequantify()
    ds.to_netcdf(filename, unlimited_dims=['sounding'])
