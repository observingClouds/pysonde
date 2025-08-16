"""
Thermodynamic functions
"""

import metpy
import metpy.calc as mpcalc
import numpy as np
import pint
import pint_pandas as pp
import xarray as xr
from moist_thermodynamics import saturation_vapor_pressures as mtsvp

# ureg = pint.UnitRegistry()
# ureg.define("fraction = [] = frac")
# ureg.define("percent = 1e-2 frac = pct")
# pp.PintType.ureg = ureg


def convert_rh_to_dewpoint(temperature, rh):
    """
    Convert T and RH to dewpoint exactly
    following the formula used by the Vaisala
    M41 sounding system

    Input
    -----
    temperature : array
        temperature in Kevlin or supported quantity
    """
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    else:
        temperature_K = temperature.data.magnitude
    if isinstance(rh, pp.pint_array.PintArray):
        rh = rh.quantity.to("percent").magnitude
    else:
        rh = rh.data.magnitude
    assert np.any(temperature_K > 100), "Temperature seems to be not given in Kelvin"
    kelvin = 15 * np.log(100 / rh) - 2 * (temperature_K - 273.15) + 2711.5
    t_dew = temperature_K * 2 * kelvin / (temperature_K * np.log(100 / rh) + 2 * kelvin)
    if isinstance(temperature, pp.pint_array.PintArray):
        t_dew = t_dew * ureg("K")
    return t_dew


def calc_saturation_pressure(temperature, method="hardy1998"):
    """
    Calculate saturation water vapor pressure.

    Parameters
    ----------
    temperature : array, pint.Quantity, or xarray.DataArray
        Temperature in Kelvin or dew point temperature for actual vapor pressure.
    method : str, optional
        Method for calculating saturation pressure:
            'hardy1998' : ITS-90 formulation (default).
            'wagner_pruss' : IAPWS-95 formulation from Wagner & Pruss (2002).

    Returns
    -------
    e_sw : array, pint.Quantity, or xarray.DataArray
        Saturation vapor pressure in Pascals.

    Examples
    --------
    >>> calc_saturation_pressure([273.15])
    array([ 611.2129107])
    >>> calc_saturation_pressure([273.15, 293.15, 253.15])
    array([  611.2129107 ,  2339.26239586,   125.58350529])
    >>> calc_saturation_pressure([273.15, 293.15, 253.15], method='wagner_pruss')
    array([611.2128459, 2339.1937366, 125.6039870])
    """
    # Ensure temperature is in Kelvin
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to("K").magnitude
    elif isinstance(temperature, xr.core.dataarray.DataArray) and hasattr(
        temperature.data, "_units"
    ):
        temperature_K = temperature.pint.to("K").metpy.magnitude
    else:
        temperature_K = temperature  # Assume already in Kelvin

    if method == "hardy1998":
        g = np.array(
            [
                -2.8365744e3,
                -6.028076559e3,
                1.954263612e1,
                -2.737830188e-2,
                1.6261698e-5,
                7.0229056e-10,
                -1.8680009e-13,
                2.7150305,
            ]
        )

        e_sw = np.exp(
            np.sum([g[i] * temperature_K ** (i - 2) for i in range(7)], axis=0)
            + g[7] * np.log(temperature_K)
        )

    elif method == "wagner_pruss":
        e_sw = mtsvp.liq_wagner_pruss(temperature_K)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Available methods: 'hardy1998', 'wagner_pruss'."
        )

    # Convert result back to PintArray or xarray DataArray
    if isinstance(temperature, pp.pint_array.PintArray):
        pp.PintType.ureg = ureg
        e_sw = pp.PintArray(e_sw, dtype="Pa")
    elif isinstance(temperature, xr.DataArray) and hasattr(temperature.data, "_units"):
        e_sw = xr.DataArray(
            e_sw, dims=temperature.dims, coords=temperature.coords
        ) * metpy.units.units("Pa")

    return e_sw


def calc_wv_mixing_ratio(total_pressure, vapor_pressure):
    """
    Calculate water vapor mixing ratio

    Parameters
    ----------
    total_pressure : DataArray or pint.Quantity
        Atmospheric pressure with units.
    vapor_pressure : DataArray or pint.Quantity
        Vapor pressure with units.

    Returns
    -------
    DataArray or Quantity
        Mixing ratio in kg/kg.
    """
    ureg = pint.get_application_registry()
    epsilon = metpy.constants.molecular_weight_ratio

    # Ensure both are pint.Quantity or pint-xarray with units of Pa
    if hasattr(vapor_pressure, "pint"):
        vapor_pressure = vapor_pressure.pint.to("Pa")
    else:
        vapor_pressure = vapor_pressure * ureg.Pa

    if hasattr(total_pressure, "pint"):
        total_pressure = total_pressure.pint.to("Pa")
    else:
        total_pressure = total_pressure * ureg.Pa

    # Compute mixing ratio
    wv_mix_ratio = epsilon * vapor_pressure / (total_pressure - vapor_pressure)

    # Ensure units are correct
    try:
        wv_mix_ratio = wv_mix_ratio.pint.to("kg/kg")
    except AttributeError:
        pass  # Return as-is if units are missing

    return wv_mix_ratio


def calc_theta_from_T(T, p):
    """
    Input :
        T : temperature
        p : pressure
    Output :
        theta : Potential temperature values
    Function to estimate potential temperature from the
    temperature and pressure in the given dataset. This function uses MetPy's
    functions to get theta:
    (i) mpcalc.potential_temperature()

    """
    theta = mpcalc.potential_temperature(p.metpy.quantify(), T.metpy.quantify())

    return theta


def calc_T_from_theta(theta, p):
    """
    Input :
        theta : potential temperature (K)
        p : pressure (hPa)
    Output :
        T : Temperature values
    Function to estimate temperature from potential temperature and pressure,
    in the given dataset. This function uses MetPy's
    functions to get T:
    (i) mpcalc.temperature_from_potential_temperature()

    """
    T = mpcalc.temperature_from_potential_temperature(
        p,
        theta,
    )

    return T
