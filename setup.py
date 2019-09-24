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
VERSION = "2.3"


with open("requirements.txt") as f:
    requirements = [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    license="GPL3",
    author="Nikos E Kouvaris",
    author_email="nkouba@gmail.com",
    description="multiNetX",
    long_description=long_description,
    url="https://github.com/nkoub/multinetx",
    keywords=["multiplex", "multilayer", "multinetx", "networkx"],
    package_dir={NAME: NAME},
    packages=find_packages(),
    setup_requires=["pytest", "pytest-runner"],
    install_requires=requirements,
    include_package_data =  True,
)
