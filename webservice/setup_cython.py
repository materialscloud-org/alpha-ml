from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import numpy as np

ext_modules = [
        Extension(
        "test",
        ["test.pyx"],
        libraries=["m"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
        language=['C'],
    ),
]

os.environ['CFLAGS'] = '-pthread -DNDEBUG -fwrapv -Ofast -Wall -Wstrict-prototypes -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fdebug-prefix-map=/build/python2.7-nbjU53/python2.7-2.7.15~rc1=. -fstack-protector-strong -Wformat -Werror=format-security -fPIC -I/usr/include/python2.7'

setup(
    name='Cython routines for kernel evaluation',
    ext_modules=cythonize(ext_modules),
)
