# Visualization of level 1 radiosonde data

This folder contains a plotting script for converted level 1 sounding data

Requirements: The script needs the `metpy` package (`conda install metpy`).

Running the script:
```
python plot_radiosonde_data.py -i infile -o outdir
```
* `infile`: input file. It should include the direction (ascent or descent) in its name. If no direction is given in the name of the input file, no SkewT plot will be created.
* `outdir`: output directory. Three subfolders `Quantities`, `SkewT`, and `Trajectories` will be created in the output directory if not already existing.

The script produces 3 plots:
 * Trajectory of the radiosonde in `Trajectories`
 * Temperature T, dew point tau, relative humidity rh, water vapor mixing ratio mr, wind speed, and wind direction in `Quantities`
 * SkewT diagram in `SkewT` (only if the direction, i.e. ascent or descent, is given in the input file name)