# Visualization of radiosonde data

This folder contains a plotting script for converted sounding data

## plot_radiosonde_data

This script needs the `metpy` package installed (`conda install metpy`).

Running the script:
```
python plot_radiosonde_data.py -i inputfilename -o outputdirectory
```
For inputfilenames of the form "{ship}\_{date}\_{time}\_{direction}.nc" (e.g. MARIA_S_MERIAN_20221102_110535_ascent.nc),
the required inputfilename is "{ship}\_{date}\_{time}" (e.g. MARIA_S_MERIAN_20221102_110535).

This script produces 5 plots:
 * Trajectory of the radiosonde
 * Wind speed and direction of ascent and descent
 * temperature T, dew point tau, relative humidity rh, and water vapor mixing ratio for ascent and descent
 * SkewT diagram of ascent
 * SkewT diagram of descent