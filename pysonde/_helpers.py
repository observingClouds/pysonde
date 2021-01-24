import inspect
import logging
import platform
from pathlib import Path, PureWindowsPath

from omegaconf import OmegaConf


class ReaderNotImplemented(Exception):
    pass


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
    configs = {}
    for config, path in config_dict.items():
        configs[config] = OmegaConf.load(path)
    cfg = OmegaConf.merge(configs)
    return cfg
