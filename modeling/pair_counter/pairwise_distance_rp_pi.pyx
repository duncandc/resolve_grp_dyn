"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.sparse import coo_matrix
cimport numpy as cnp
cimport cython 
from libc.math cimport ceil, sqrt
from libcpp.vector cimport vector

__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('pairwise_distance_rp_pi', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_distance_rp_pi(x1in, y1in, z1in, x2in, y2in, z2in, rp_max, pi_max):
    """ 
    Cython engine for returning pairs of points and perendicular and parallel 
    seperation with an observer at (0,0)
    
    Parameters 
    ------------
    x1in, y1in, z1in : arrays 
        Numpy arrays storing Cartesian coordinates of points in sample 1
    
    x2in, y2in, z2in : arrays 
        Numpy arrays storing Cartesian coordinates of points in sample 2
    
    rp_max : array
        maximum perpendicular separation distance to search for and return pairs
    
    pi_max : array
        maximum parallel separation distance to search for and return pairs
    
    Returns 
    --------
    rp_distance : numpy.array
        array of perpednicular pairwise separation distances
    
    pi_distance : numpy.array
        array of parallel pairwise separation distances
    
    i : numpy.array
        array of 0-indexed indices in sample 1
    
    j : numpy.array
        array of 0-indexed indices in sample2
    
    """
    
    cdef cnp.float64_t rp_max_p = rp_max
    cdef cnp.float64_t pi_max_p = pi_max
    
    cdef cnp.float64_t[:] x1 = np.ascontiguousarray(x1in, dtype=np.float64)
    cdef cnp.float64_t[:] y1 = np.ascontiguousarray(y1in, dtype=np.float64)
    cdef cnp.float64_t[:] z1 = np.ascontiguousarray(z1in, dtype=np.float64)
    cdef cnp.float64_t[:] x2 = np.ascontiguousarray(x2in, dtype=np.float64)
    cdef cnp.float64_t[:] y2 = np.ascontiguousarray(y2in, dtype=np.float64)
    cdef cnp.float64_t[:] z2 = np.ascontiguousarray(z2in, dtype=np.float64)
    
    cdef vector[cnp.int_t] i_ind
    cdef vector[cnp.int_t] j_ind
    cdef vector[cnp.float64_t] rp_distances, pi_distances
    
    cdef cnp.float64_t sx, sy, sz, lx, ly, lz, l2, ll, spar, s2, sperp 
    cdef int Ni, Nj, i, j
    
    Ni = len(x1)
    Nj = len(x2)
    
    #loop over points in cell1 points
    for i in range(0,Ni):
        #loop over points in cell2 points
        for j in range(0,Nj):
            #calculate the square distance
            sx = x1[i] - x2[j]
            sy = y1[i] - y2[j]
            sz = z1[i] - z2[j]
            lx = 0.5*(x1[i] + x2[j])
            ly = 0.5*(y1[i] + y2[j])
            lz = 0.5*(z1[i] + z2[j])
            l2 = (lx * lx) + (ly * ly) + (lz * lz)
            ll = sqrt(l2)
            spar = abs(((sx * lx) + (sy * ly) + (sz * lz)) / ll)
            s2 = (sx * sx) + (sy * sy) + (sz * sz)
            sperp = sqrt(s2 - spar * spar)
            
            if (spar <= pi_max_p) & (sperp <= rp_max_p):
                rp_distances.push_back(sperp)
                pi_distances.push_back(spar)
                i_ind.push_back(i)
                j_ind.push_back(j)
    
    rp = np.array(rp_distances).astype(float)
    pi = np.array(pi_distances).astype(float)
    i_inds = np.array(i_ind)
    j_inds = np.array(j_ind)
    
    #rp_distances = coo_matrix((rp, (i_inds, j_inds)), shape=(Ni,Nj))
    #pi_distances = coo_matrix((pi, (i_inds, j_inds)), shape=(Ni,Nj))
    
    return rp_distances, pi_distances, i_inds, j_inds
