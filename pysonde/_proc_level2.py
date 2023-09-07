import logging
import os
from pathlib import Path

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from metpy.units import units
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from . import _helpers as h
from . import meteorology_helpers as mh
from . import thermodynamics as td
from .readers.readers import pysondeL1


def prepare_data_for_interpolation(ds, uni, variables, reader=pysondeL1):
    u, v = mh.get_wind_components(ds.wdir, ds.wspd)
    ds["u"] = xr.DataArray(u.data, dims=["level"])
    ds["v"] = xr.DataArray(v.data, dims=["level"])

    if "alt_WGS84" in ds.keys():
        # Convert lat, lon, alt to cartesian coordinates
        ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
        lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
        x, y, z = pyproj.transform(
            lla,
            ecef,
            ds.lon.values,
            ds.lat.values,
            ds["alt_WGS84"].values,
            radians=False,
        )
        for var, val in {"x": x, "y": y, "z": z}.items():
            ds[var] = xr.DataArray(val, dims=["level"])
    else:
        logging.warning(
            "No WGS84 altitude could be found. The averaging of the position might be faulty especially at the 0 meridian and close to the poles"
        )

    # Calculate more thermodynamical variables (used for interpolation)
    td.metpy.units.units = uni
    theta = td.calc_theta_from_T(ds["ta"], ds["p"])
    e_s = td.calc_saturation_pressure(ds["ta"])
    w_s = mpcalc.mixing_ratio(e_s, ds["p"].metpy.quantify())
    w = ds["rh"].data * w_s
    q = w / (1 + w)

    w["level"] = ds.alt.data
    w = w.rename({"level": "alt"})
    w = w.expand_dims({"sounding": 1})
    q["level"] = ds.alt.data
    q = q.rename({"level": "alt"})
    q = q.expand_dims({"sounding": 1})
    theta["level"] = ds.alt.data
    theta = theta.rename({"level": "alt"})
    theta = theta.expand_dims({"sounding": 1})

    ds = ds.rename_vars({"alt": "altitude"})
    ds = ds.rename({"level": "alt"})
    ds["alt"] = ds.altitude.data
    ds = ds.reset_coords()
    ds = ds.expand_dims({"sounding": 1})

    ds_new = xr.Dataset()  # ds.copy()
    ds_new["mr"] = w.reset_coords(drop=True)
    ds_new["theta"] = theta.reset_coords(drop=True)
    ds_new["specific_humidity"] = q.reset_coords(drop=True)

    for var in ds.data_vars:
        if var not in ds_new.data_vars and var not in ds_new.coords:
            try:
                ds_new[var] = ds[var]
            except NameError:
                logging.warning(f"Variable {var} not found.")
                pass

    for variable_name_in, variable_name_out in variables:
        try:
            ds_new = ds_new.rename({variable_name_in: variable_name_out})
            ds_new[variable_name_out].attrs = ds[
                variable_name_in
            ].attrs  # Copy attributes from input
        except (ValueError, KeyError):
            logging.warning(f"Variable {variable_name_in} not found.")
            pass

    return ds, ds_new


def interpolation(ds_new, method, interpolation_grid, sounding, variables, cfg):
    if method == "linear":
        # Druck logarithmisch interpolieren

        pres_int_p = ds_new.pressure.pint.to("hPa").values[0]
        pres_int_a = ds_new.altitude.pint.to("m").values[0]

        ds_new = ds_new.dropna(dim="alt", how="any")
        # subset=output_variables,
        ds_interp = ds_new.interp(alt=interpolation_grid)

        # Logarithmic Pressure Interpolation
        # """
        dims_1d = ["alt"]
        coords_1d = {"alt": ds_interp.alt.data}

        alt_out = ds_interp.alt.values

        interp_pres = mh.pressure_interpolation(
            pres_int_p, pres_int_a, alt_out
        ) * sounding.unitregistry("hPa")

        ds_interp["pressure"] = xr.DataArray(
            interp_pres, dims=dims_1d, coords=coords_1d
        )
        ds_interp["pressure"] = ds_interp["pressure"].expand_dims({"sounding": 1})
        # """

        for var_in, var_out in variables:
            try:
                ds_interp[var_out] = ds_interp[var_out].pint.quantify(
                    ds_new[var_out].pint.units
                )
                ds_interp[var_out] = ds_interp[var_out].pint.to(
                    cfg.level2.variables[var_in].attrs.units
                )
            except (KeyError, ValueError, ConfigAttributeError) as e:
                logging.warning(
                    f"Likely no unit has been found for {var_out}, raising {e}"
                )
                pass

    elif method == "bin":
        interpolation_bins = np.arange(
            cfg.level2.setup.interpolation_grid_min
            - cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_max
            + cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_inc,
        )
        # Workaround for issue https://github.com/pydata/xarray/issues/6995
        ds_new["flight_time"] = ds_new.flight_time.astype(int)
        ds_interp = ds_new.groupby_bins(
            "altitude",
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        ).mean()
        ds_interp = ds_interp.transpose()
        ds_interp = ds_interp.rename({"altitude_bins": "alt"})

        # Create bounds variable
        ds_interp["alt_bnds"] = xr.DataArray(
            np.array([interpolation_bins[:-1], interpolation_bins[1:]]).T,
            dims=["alt", "nv"],
            coords={"alt": ds_interp.alt.data},
        )

        ds_interp["launch_time"] = ds_new["launch_time"]

        ## Interpolation NaN
        units = {v: ds_interp[v].pint.units for v in ds_interp.data_vars}
        ds_interp = ds_interp.interpolate_na(
            "alt", max_gap=cfg.level2.setup.max_gap_fill, use_coordinate=True
        )
        ds_interp = ds_interp.pint.quantify(
            units
        )  # pint.interpolate_na does not support max_gap yet and looses units

        ds_interp["flight_time"] = ds_interp.flight_time.astype("datetime64[ns]")

    return ds_interp


def adjust_ds_after_interpolation(ds_interp, ds, ds_input, variables, cfg):
    dims_2d = ["sounding", "alt"]
    dims_1d = ["alt"]
    ureg = ds["lat"].pint.units._REGISTRY
    coords_1d = {"alt": ds_interp.alt.pint.quantify("m", unit_registry=ureg)}

    wind_u = ds_interp.isel({"sounding": 0})["wind_u"]
    wind_v = ds_interp.isel({"sounding": 0})["wind_v"]
    dir, wsp = mh.get_directional_wind(wind_u, wind_v)

    ds_interp["wind_direction"] = xr.DataArray(
        dir.expand_dims({"sounding": 1}).data, dims=dims_2d, coords=coords_1d
    )
    ds_interp["wind_speed"] = xr.DataArray(
        wsp.expand_dims({"sounding": 1}).data, dims=dims_2d, coords=coords_1d
    )

    if "alt_WGS84" in ds.keys():
        ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
        lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
        lon, lat, alt = pyproj.transform(
            ecef,
            lla,
            ds_interp["x"].values,
            ds_interp["y"].values,
            ds_interp["z"].values,
            radians=False,
        )

        for var, val in {
            "lat": lat,
            "lon": lon,
            "alt_WGS84": alt,
        }.items():
            try:
                ds_interp[var] = xr.DataArray(
                    val, dims=dims_1d, coords=coords_1d
                ).pint.quantify(
                    cfg.level2.variables[var].attrs.units, unit_registry=ureg
                )
            except ConfigKeyError:
                pass
        del ds_interp["x"]
        del ds_interp["y"]
        del ds_interp["z"]
        del ds_interp["alt_WGS84"]

    ds_input = ds_input.sortby("alt")
    ds_input.alt.load()
    ds_input.p.load()
    ds_input = ds_input.reset_coords()

    # ds_interp['pressure'] = ds_interp['pressure'].pint.to(cfg.level2.variables['p'].attrs.units)
    # ds_interp['pressure'] = ds_interp['pressure'].expand_dims({'sounding':1})

    ds_interp["launch_time"] = xr.DataArray(
        [ds_interp.isel({"sounding": 0}).launch_time.item() / 1e9], dims=["sounding"]
    )

    # Calculations after interpolation
    # Recalculate temperature and relative humidity from theta and q

    temperature = td.calc_T_from_theta(
        ds_interp.isel(sounding=0)["theta"].pint.to("K"),
        ds_interp.isel(sounding=0)["pressure"].pint.to("hPa"),
    )

    ds_interp["temperature"] = xr.DataArray(
        temperature.data, dims=dims_1d, coords=coords_1d
    )
    ds_interp["temperature"] = ds_interp["temperature"].expand_dims({"sounding": 1})

    w = (ds_interp.isel(sounding=0)["specific_humidity"]) / (
        1 - ds_interp.isel(sounding=0)["specific_humidity"]
    )
    e_s = td.calc_saturation_pressure(ds_interp.isel(sounding=0)["temperature"])
    w_s = mpcalc.mixing_ratio(e_s, ds_interp.isel(sounding=0)["pressure"].data)
    relative_humidity = w / w_s * 100

    ds_interp["relative_humidity"] = xr.DataArray(
        relative_humidity.data, dims=dims_1d, coords=coords_1d
    )
    ds_interp["relative_humidity"] = ds_interp["relative_humidity"].expand_dims(
        {"sounding": 1}
    )

    ds_interp["relative_humidity"].data = ds_interp["relative_humidity"].data * units(
        "%"
    )

    dewpoint = td.convert_rh_to_dewpoint(
        ds_interp.isel(sounding=0)["temperature"],
        ds_interp.isel(sounding=0)["relative_humidity"],
    )

    ds_interp["dewpoint"] = xr.DataArray(dewpoint.data, dims=dims_1d, coords=coords_1d)
    ds_interp["dewpoint"] = ds_interp["dewpoint"].expand_dims({"sounding": 1})
    ds_interp["dewpoint"].data = ds_interp["dewpoint"].data * units.K
    # ds_interp = ds_interp.drop('dew_point')

    # ds_interp = ds_interp.drop('altitude')

    ds_interp["mixing_ratio"].data = ds_interp["mixing_ratio"].data * units("g/g")
    ds_interp["specific_humidity"].data = ds_interp["specific_humidity"].data * units(
        "g/g"
    )

    return ds_interp


def count_number_of_measurement_within_bin(ds_interp, ds_new, cfg, interpolation_grid):
    interpolation_bins = np.arange(
        cfg.level2.setup.interpolation_grid_min
        - cfg.level2.setup.interpolation_grid_inc / 2,
        cfg.level2.setup.interpolation_grid_max
        + cfg.level2.setup.interpolation_grid_inc / 2,
        cfg.level2.setup.interpolation_grid_inc,
    )

    # Count number of measurements within each bin
    dims_2d = ["sounding", "alt"]
    coords_1d = {"alt": ds_interp.alt}

    ds_interp["N_ptu"] = xr.DataArray(
        ds_new.pressure.groupby_bins(
            "alt",
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        )
        .count()
        .values,
        dims=dims_2d,
        coords=coords_1d,
    )
    ds_interp["N_gps"] = xr.DataArray(
        ds_new.latitude.groupby_bins(
            "alt",
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        )
        .count()
        .values,
        dims=dims_2d,
        coords=coords_1d,
    )

    # Cell method used
    data_exists = np.where(np.isnan(ds_interp.isel(sounding=0).pressure), False, True)
    data_mean = np.where(
        np.isnan(ds_interp.isel(sounding=0).N_ptu), False, True
    )  # no data or interp: nan; mean > 0
    data_method = np.zeros_like(data_exists, dtype="uint")
    data_method[np.logical_and(data_exists, data_mean)] = 2
    data_method[np.logical_and(data_exists, ~data_mean)] = 1
    ds_interp["m_ptu"] = xr.DataArray([data_method], dims=dims_2d, coords=coords_1d)
    ds_interp["N_ptu"].values[0, np.logical_and(~data_mean, data_method > 0)] = 0

    data_exists = np.where(np.isnan(ds_interp.isel(sounding=0).latitude), False, True)
    data_mean = np.where(
        np.isnan(ds_interp.isel(sounding=0).N_gps), False, True
    )  # no data or interp: nan; mean > 0
    data_method = np.zeros_like(data_exists, dtype="uint")
    data_method[np.logical_and(data_exists, data_mean)] = 2
    data_method[np.logical_and(data_exists, ~data_mean)] = 1
    ds_interp["m_gps"] = xr.DataArray([data_method], dims=dims_2d, coords=coords_1d)
    ds_interp["N_gps"].values[0, np.logical_and(~data_mean, data_method > 0)] = 0

    return ds_interp


def finalize_attrs(ds_interp, ds, cfg, file, variables):
    import pandas as pd
    from netCDF4 import num2date

    def convert_num2_date_with_nan(num, format):
        if not np.isnan(num):
            return num2date(
                num,
                format,
                only_use_python_datetimes=True,
                only_use_cftime_datetimes=False,
            )
        else:
            return pd.NaT

    convert_nums2date = np.vectorize(convert_num2_date_with_nan)

    # ds_interp['flight_time'].data = convert_nums2date(ds_interp.flight_time.data, "seconds since 1970-01-01")
    ds_interp["launch_time"].data = convert_nums2date(
        ds_interp.launch_time.data, "seconds since 1970-01-01"
    )

    # direction = get_direction(ds_interp, ds)
    most_common_vertical_movement = np.argmax(
        [
            np.count_nonzero(ds_interp.ascent_rate > 0),
            np.count_nonzero(ds_interp.ascent_rate < 0),
        ]
    )
    ds_interp["ascent_flag"] = xr.DataArray(
        [most_common_vertical_movement], dims=["sounding"]
    )

    # # Copy trajectory id from level1 dataset
    # ds_interp['sounding'] = xr.DataArray([ds['sounding'].values])#, dims=['sounding'])
    ds_interp.sounding.attrs = ds["sounding"].attrs

    # merged_conf = OmegaConf.merge(config.level2, meta_data_cfg)
    # merged_conf._set_parent(OmegaConf.merge(config, meta_data_cfg))
    ds_interp.attrs = ds.attrs

    ds_interp = h.replace_global_attributes(ds_interp, cfg)
    ds_interp.attrs["source"] = str(file).split("/")[-1]

    ds_interp = h.set_additional_var_attributes(
        ds_interp, cfg.level2.variables, variables
    )

    # Transpose dataset if necessary
    for variable in ds_interp.data_vars:
        if variable == "alt_bnds":
            continue
        dims = ds_interp[variable].dims
        if (len(dims) == 2) and (dims[0] != "sounding"):
            ds_interp[variable] = ds_interp[variable].T

    # time_dt = pd.Timestamp(np.datetime64(ds_interp.isel({'sounding': 0}).launch_time.data.astype("<M8[ns]")))

    return ds_interp


def export(output_fmt, ds_interp, cfg):
    """Saves sounding to disk"""

    if ds_interp.ascent_flag.values[0] == 0:
        direction = "AscentProfile"
    elif ds_interp.ascent_flag.values[0] == 1:
        direction = "DescentProfile"

    # time_fmt = time_dt.strftime('%Y%m%dT%H%M')
    outfile = output_fmt.format(
        platform=cfg.main.platform,
        campaign=cfg.main.campaign,
        campaign_id=cfg.main.campaign_id,
        direction=direction,
        version=cfg.main.data_version,
        level="2",
    )
    launch_time = pd.to_datetime(ds_interp.launch_time.item(0))
    outfile = launch_time.strftime(outfile)
    directory = os.path.dirname(outfile)
    Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Write output to {}".format(outfile))
    h.write_dataset(ds_interp, outfile)
