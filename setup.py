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
    licence="GPL3",
    author="Nikos E Kouvaris",
    author_email="nkouba@gmail.com",
    description="multiNetX",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/nkoub/multinetx",
    keywords=["multiplex", "multilayer", "multinetx", "networkx"],
    # package_dir={NAME: NAME},
    packages=find_packages(),
    setup_requires=["pytest", "pytest-runner"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
