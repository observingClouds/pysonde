import inspect
import logging
import platform
import subprocess as sp
import time
from pathlib import Path, PureWindowsPath

from omegaconf import OmegaConf


class ReaderNotImplemented(Exception):
    pass


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


def replace_placeholders_cfg(cfg, subset="level1"):
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
    if "version" in cfg[subset].keys():
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
