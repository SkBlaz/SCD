
## Py3plex installation file. Cython code for fa2 is the courtesy of Bhargav Chippada.
## https://github.com/bhargavchippada/forceatlas2

from os import path
import sys
from setuptools import setup,find_packages
from setuptools.extension import Extension
import argparse
    
setup(name='SCD',
      version='0.01',
      description="Embedding-based Silhouette Community Detection",
      url='https://github.com/SkBlaz/SCD',
      python_requires='>3.6.0',
      author='Blaž Škrlj',
      author_email='blaz.skrlj@ijs.si',
      license='bsd-3-clause-clear',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True)
