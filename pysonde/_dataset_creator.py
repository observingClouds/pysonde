"""Helper to create dataset

Create an empty dataset based on a template.yaml file
in the format of

    global_attrs:
        title: "Awesome dataset"
        description: "It is just awesome"
    coordinates:
        time:
            attrs:
                units: "seconds since 1970-01-01 00:00:00"
                calendar: "standard"
                axis: "T"
                standard_name: "time"
            dimension: ${time_dimension}
        range:
            attrs:
                units: "m"
            dimension: ${range_dimension}
    variables:
        label:
            attrs:
                description: "really great variable"
            coordinates:
                - time
                - range

The dimensions need to be passed during runtime. This can be easily
dome by creating an additional OmegaConf instance and merge it with
the one above, e.g.
    runtime_conf = OmegaConf.create({'time_dimension':100, 'range_dimension':1000})
    cfg = OmegaConf.merge(template_conf, runtime_conf)
    ds = create_dataset(cfg)

"""
import logging

import xarray as xr


def create_dataset(cfg):
    ds = xr.Dataset()
    ds = set_global_attrs(cfg, ds)
    ds = set_coords(cfg, ds)
    ds = set_variables(cfg, ds)
    return ds


def set_global_attrs(cfg, ds):
    logging.debug("Add global attributes")
    if "global_attrs" in cfg.keys():
        ds.attrs = cfg["global_attrs"]
    return ds


def set_coords(cfg, ds):
    if "coordinates" in cfg.keys():
        for coord, params in cfg.coordinates.items():
            if type(params["dimension"]) == int:
                ds = ds.assign_coords(
                    {coord: range(params["dimension"])}
                )  # write temporary values to coord
            else:
                ds = ds.assign_coords({coord: params["dimension"]})
            if "attrs" in params.keys():
                ds[coord].attrs = params["attrs"]
    return ds


def set_variables(cfg, ds):
    logging.debug("Add variables to dataset")
    if "variables" in cfg.keys():
        for var, params in cfg.variables.items():
            coord_dict = {coord: ds[coord] for coord in params.coordinates}
            ds[var] = xr.DataArray(None, coords=coord_dict, dims=params.coordinates)
            if "attrs" in params.keys():
                ds[var].attrs = params["attrs"]
    return ds
