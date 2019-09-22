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
VERSION = "2.1"

setup(
    name=NAME,
    version=VERSION,
    description="multiNetX",
    licence="GPL3",
    __author__ = "Nikos E. Kouvaris <nkouba@gmail.com>",
    copyright="Copyright (C) 2013-2019 by Nikos E. Kouvaris <nkouba@gmail.com>."
              "Project LASAGNE -- multi-LAyer SpAtiotemporal Generalized NEtworks",
    author="Nikos E Kouvaris",
    author_email="nkouba@gmail.com",
    url="https://github.com/nkoub/multinetx",
    keywords=["multiplex", "multilayer", "multinetx", "networkx"],
    package_dir={NAME: NAME},
    packages=find_packages(),
    setup_requires=["pytest", "pytest-runner"],
    long_description="multiNetX is a python package for the manipulation and visualization of multilayer networks."
                     "It is build on NetworkX"
)
