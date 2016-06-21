#!/usr/bin/python

#Author: Duncan Campbell
#June, 2016

"""
Model the RESOLVE data using a BMHM relation
"""

###packages###
from __future__ import ( division, print_function, absolute_import, unicode_literals)
import numpy as np
import h5py
from astropy.table import Table
from scipy.stats import rankdata
import halotools
from halotools.utils import group_member_generator
import matplotlib.pyplot as plt


def main():
    
    N=10**8
    pos = np.random.random((N,3))*250.0
    pos = pos-250.0/2.0
    vels = pos*0.0
    
    #put into ra-dec-z
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    from halotools.mock_observables import ra_dec_z
    ra,dec,z = ra_dec_z(pos, vels, cosmo=cosmo)
    ra = np.degrees(ra)
    dec= np.degrees(dec)
    
    #make ra,dec cuts
    #RESOLVE B
    resolve_b_ra_mask = ((22.0*(360.0/24.0)-180.0) < ra) | (ra < (3.0*(360.0/24.0)-180.0))
    resolve_b_dec_mask = (-1.25 < dec) & (dec < 1.25)
    #RESOLVE B
    resolve_a_ra_mask = ((131.25-180.0) < ra) & (ra < (236.25-180.0))
    resolve_a_dec_mask = (0.0 < dec) & (dec < 5.0)
    
    #make redshift cut
    from astropy.constants import c
    speed_of_light = c.to('km/s').value
    resolve_z_mask = (4500.0 < (z * speed_of_light)) & ((z * speed_of_light) < 7000.0)
    
    #make combined mask
    resolve_b_mask = resolve_b_ra_mask & resolve_b_dec_mask & resolve_z_mask
    resolve_a_mask = resolve_a_ra_mask & resolve_a_dec_mask & resolve_z_mask
    N_in_resolve_a = np.sum(resolve_a_mask)
    N_in_resolve_b = np.sum(resolve_b_mask)
    N_in_resolve = N_in_resolve_a + N_in_resolve_b
    print("number of randoms in RESOLVE a: ", N_in_resolve_a)
    print("number of randoms in RESOLVE b: ", N_in_resolve_b)
    print("number of randoms in RESOLVE: ", N_in_resolve)
    
    volume = (N_in_resolve_a/N)*250.0**3
    print("volume of RESOLVE a: {0} h-3Mpc, {1} Mpc".format(volume, volume/(0.7**3)))
    volume = (N_in_resolve_b/N)*250.0**3
    print("volume of RESOLVE b: {0} h-3Mpc, {1} Mpc".format(volume, volume/(0.7**3)))
    volume = (N_in_resolve/N)*250.0**3
    print("volume of RESOLVE a + b: {0} h-3Mpc, {1} Mpc".format(volume, volume/(0.7**3)))
    
    """
    resolve_mask = resolve_b_mask | resolve_a_mask
    
    plot_sky_dist(ra[resolve_mask],dec[resolve_mask])
    """
    

def plot_sky_dist(ra,dec):
    
    y = np.cos(np.radians(dec)-np.pi/2.0)
    x = ra
    
    new_tick_locations = np.array([-90.0,-60.0,-30.0,0.0,60.0,30.0,90.0])
    new_ticks = np.array([r'$-90^{\circ}$',r'$-60^{\circ}$',r'$-30^{\circ}$',r'$0^{\circ}$',r'$60^{\circ}$',r'$30^{\circ}$',r'$90^{\circ}$'])
    def tick_function(X):
        V = np.cos(np.radians(X)-np.pi/2.0)
        return [z for z in V]
    
    fig = plt.figure(figsize=(3.3,3.3))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left = 0.2, right = 0.9, top=0.9, bottom=0.2)
    #ax.grid(True)
    ax.scatter(x, y, s=0.2, alpha=1, lw=0, rasterized=True)
    ax.set_xlim([-180.0,180.0])
    plt.ylim([-1.0,1.0])
    #ax.set_xticks(np.arange(110,271,40))
    #ax.set_xticklabels([r'$110^{\circ}$',r'$150^{\circ}$',r'$190^{\circ}$',r'$230^{\circ}$',r'$270^{\circ}$'])
    ax.set_yticks(tick_function(new_tick_locations))
    ax.set_yticklabels(new_ticks)
    plt.xlabel(r'$\alpha ~ [{\rm J2000}]$')
    plt.ylabel(r'$\delta ~ [{\rm J2000}]$')
    plt.show()

if __name__ == '__main__':
    main()
