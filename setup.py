# -*- coding: utf-8 -*-
"""
To install the library, run the following

python setup.py install

prerequisite: setuptools
http://pypi.python.org/pypi/setuptools

Created on 22/9/19
@author: Nikos Kouvaris <nkouba@gmail.com>
Licence, GPL
"""


from setuptools import setup, find_packages

NAME = "multinetx"
VERSION = "3.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    description="multiNetX",
    licence="GPL3",
    copyright="Copyright (C) 2013-2019 by Nikos E. Kouvaris <nkouba@gmail.com>",
    author="Nikos E Kouvaris",
    author_email="nkouba@gmail.com",
    url="https://github.com/nkoub/multinetx",
    keywords=["multiplex", "multilayer", "multinetx", "networkx"],
    package_dir={NAME: NAME},
    packages=find_packages(),
    setup_requires=["pytest", "pytest-runner"],
    include_package_data=True,
    long_description_content_type='text/markdown',
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
