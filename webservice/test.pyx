cimport cython # the main Cython stuff

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.parallel cimport prange

cimport scipy.special.cython_special as csc # Cython interfaces to Scipy Special functions (sph_harm, gamma, hyp1f1)

from libc.math cimport sqrt,abs
from libc.stdio cimport printf

cdef extern void zero_array (double *array, double value, size_t n)

# NumPy and Scipy Special Functions
import sys
import numpy as np
import scipy.special as sc

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _testf(long lam,long nat1,long nat2,long featsize,long featsize0,complex[:,:,:] ps1,complex[:,:,:,:] ps2,complex[:,:] ps01, complex [:,:,:] ps02,complex klocal,complex klocal0,complex[:,:,:,:] kernel):
    cdef Py_ssize_t i,j,iat,jat,k,mu,nu

    for iat in xrange(nat1):
        for jat in xrange(nat2):
            for k in xrange(featsize0):
                klocal0 = klocal0 + ps01[iat,k]*(ps02[jat,0,k].conjugate())
            for mu in xrange(2*lam+1):
                for nu in xrange(2*lam+1):
                    for k in xrange(featsize):
                        klocal = klocal + ps1[iat,mu,k] * (ps2[jat,0,nu,k].conjugate())
                    kernel[0,jat,mu,nu] = kernel[0,jat,mu,nu] + klocal * klocal0
                    klocal = 0.0 + 0.0j
            klocal0 = 0.0 + 0.0j

#                                   for mu in xrange(2*lam+1):
#                       for nu in xrange(2*lam+1):
#                           for k in xrange(featsize):
#                               klocal = klocal + ps1[i,iat,mu,k]*(ps2[j,jat,nu,k].conjugate())
#                           kernel[i,j,mu,nu] =  kernel[i,j,mu,nu] + klocal * klocal0**(zeta-1)
#                           klocal = 0.0 + 0.0j
#                   klocal0 = 0.0 + 0.0j

def test_func(lam,nat1,nat2,featsize,featsize0,ps1,ps2,ps01,ps02):
    kernel  = np.zeros((1,nat2,2*lam+1,2*lam+1),dtype=complex)
    klocal  = complex(0.0,0.0)
    klocal0 = complex(0.0,0.0)
    _testf(lam,nat1,nat2,featsize,featsize0,ps1,ps2,ps01,ps02,klocal,klocal0,kernel)
    return kernel
