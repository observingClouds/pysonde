#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup the package pysonde."""

from setuptools import find_packages, setup

import versioneer

_TEST_REQUIRES = [
    "mock",
    "pytest",
    "pytest-pycodestyle",
    "pytest-cov",
    "pytest-pydocstyle",
    "pytest-flake8",
    "pytest-isort",
    "pytest-mock",
    "pytest-pep8",
    "pytest-pylint",
    "pytest-yapf3>=0.4.0",
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
    install_requires=[],
    setup_requires=[
        "pytest-runner",
        "setuptools-scm",
    ],
    tests_require=_TEST_REQUIRES,
    extras_require={
        "dev": _TEST_REQUIRES
        + [
            "flake8",
            "isort",
            "pylint<2.3.0",
            "yapf",
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
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": [
            "sounding_converter=pysonde.pysonde:main",
            "sounding_visualize=pysonde.make_quicklooks_rs41:main",
        ]
    },
)
