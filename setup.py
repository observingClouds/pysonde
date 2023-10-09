#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup the package pysonde."""

from setuptools import find_packages, setup

import versioneer

with open("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

_TEST_REQUIRES = [
    "mock",
    "pytest",
    "pytest-cov",
    "pytest-lazy-fixture",
    "flake8",
    "isort",
    "black",
    "mypy",
]

setup(
    # Metadata
    name="pysonde",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Hauke Schulz",
    author_email="",
    url="https://github.com/observingClouds/pysonde",
    long_description="Post-processing tool for atmospheric sounding data",
    # Package modules and data
    packages=find_packages(),
    # Requires
    python_requires=">=3.5",
    install_requires=requirements,
    setup_requires=[
        "pytest-runner",
        "setuptools-scm",
        "setuptools>=30.3.0",
    ],
    include_package_data=True,
    tests_require=_TEST_REQUIRES,
    extras_require={
        "dev": _TEST_REQUIRES
        + [
            "pylint<2.3.0",
            "yapf",
            "pre-commit",
        ],
    },
    package_data={"pysonde": []},
    # PyPI Metadata
    platforms=["any"],
    classifiers=[
        # See: https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "sounding_converter=pysonde.pysonde:main",
            "sounding_visualize=pysonde.make_quicklooks_rs41:main",
        ]
    },
)
