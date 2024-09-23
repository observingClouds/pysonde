=========
CHANGELOG
=========

unreleased
------------------

* Add zenodo release information (:pr:`24`) `Hauke Schulz`_.
* Fix warning about seeting copy of a slice of a DataFrame (:pr:`28`) `Marius Rixen`_.
* Update infrastructure to use pyproject.toml with pdm and ruff (:pr:`29`, :pr:`34`) `Hauke Schulz`_.
* Added a function computing the distance tarvelled on the ascending branch and adding it to the trajectory plot. (:pr:`33`) `Marius Winkler`_.
* Add reader for METEOMODEM radiosonde data from `.cor` files (:pr:`26`) `Hauke Schulz` and `Marius Rixen`_.
* Minor fixes of bugs introduced by aboves changes (pysonde version; sounding id; sonde type) (:pr:`38`) `Hauke Schulz`_.

0.0.5 (2023-10-19)
------------------

* Add scripts to plot radiosonde data incl. trajectory, skewT and profiles of measured quantities (:pr:`21`) `Laura Köhler`_.
* Update pre-commit linters with slight code adjustments `Hauke Schulz`_.
* bump up python to 3.11 for CI tests

0.0.4 (2022-11-02)
------------------

* Add pip install dependencies (:issue:`16`, :pr:`19`) `Hauke Schulz`_.
* Fix Pint version uncomptabilities (:issue:`17`, :pr:`18`) `Hauke Schulz`_.
* Fix missing altitude unit (`6be9b10`) `Hauke Schulz`_.

