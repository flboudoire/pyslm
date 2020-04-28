# -*- coding: utf-8 -*-

# Learn more: https://github.com/lukeparry/pyccx/setup.py
import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__),
    'pyccx/version.py')

with open(version_file, 'r') as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split('=')[-1])

# minimal requirements for installing pyccx
# note that `pip` requires setuptools itself
requirements_default = set([
    'numpy',     # all data structures
    'shapely',
    'rtree',
    'networkx',
    'trimesh',  # Required for meshing geometry
    'scikit-image',
    'setuptools'  # used for packaging
])

# "easy" requirements should install without compiling
# anything on Windows, Linux, and Mac, for Python 2.7-3.4+
requirements_easy = set([
    'setuptools',  # do setuptools stuff
    'colorlog'])   # log in pretty colors


# requirements for building documentation
requirements_docs = set([
    'sphinx',
    'jupyter',
    'sphinx_rtd_theme',
    'pypandoc',
    'autodocsumm'])

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

dev_mode = os.path.exists('/external/pyclipper/dev')


ext = Extension("pyclipper",
                sources =  ["external/pyclipper/pyclipper/pyclipper.pyx", "external/pyclipper/pyclipper/clipper.cpp"],
                language="c++",
                # define extra macro definitions that are used by clipper
                # Available definitions that can be used with pyclipper:
                # use_lines, use_int32
                # See pyclipper/clipper.hpp
                define_macros=[('use_lines', 1),
                               ('use_xyz', 1)]
                )

extB = Extension("pyslm.pyclipper",
                sources = ["external/pyclipper/pyclipper/pyclipper.cpp", "external/pyclipper/pyclipper/clipper.cpp"],
                language="c++",
                # define extra macro definitions that are used by clipper
                # Available definitions that can be used with pyclipper:
                # use_lines, use_int32
                # See pyclipper/clipper.hpp
                define_macros=[('use_lines', 1),
                               ('use_xyz', 1)]
                )

setup(
    name='PySLM',
    version=__version__,
    description=' Python Package for Additive Manufacturing Development',
    long_description_content_type='text/x-rst',
    long_description=long_description,
    author='Luke Parry',
    author_email='dev@lukeparry.uk',
    url='https://github.com/drlukeparry/pyslm',
    keywords=['3D Printing', 'AM', 'Additive Manufacturing', 'Geometry', 'SLM', 'Selective Laser Melting'],
    ext_modules=cythonize([ext,extB]),
    setup_requires=[
       'cython>=0.28'
    ],
    python_requires='>=3.5',
    classifiers=[
        'License :: OSI Approved :: LGPL License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'],
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    install_requires=list(requirements_default),
    extras_require={'easy': list(requirements_easy),
                    'docs': list(requirements_docs)}
)


