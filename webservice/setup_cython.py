from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

ext_modules = [
        Extension(
        "test",
        ["test.pyx"],
        libraries=["m"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='Cython routines for kernel evaluation',
    ext_modules=cythonize(ext_modules),
)
