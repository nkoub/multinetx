from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='multinetx',
      version=version,
      description="multiNetX is a python package for the manipulation and visualization of multilayer networks",
      long_description="""\
multiNetX is a python package for the manipulation and visualization of multilayer networks. The core of this package is a MultilayerGraph, a class that inherits all properties from networkx.Graph()""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='network networkx multiplex multilayer graph multigraph',
      author='Nikos E Kouvaris',
      author_email='nkouba@gmail.com',
      url='https://github.com/nkoub/multinetx',
      license='GPL3',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
