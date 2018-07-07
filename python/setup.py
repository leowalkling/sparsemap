import os

from setuptools.extension import Extension
from skbuild import setup

setup(
    name="sparsemap",
    version="0.1.dev1",
    author="Vlad Niculae",
    author_email="vlad@vene.ro",
    packages=['sparsemap'],
    package_dir={"sparsemap": "sparsemap"},
    include_package_data=True,
    cmake_source_dir="..",
    cmake_install_dir="."
)
