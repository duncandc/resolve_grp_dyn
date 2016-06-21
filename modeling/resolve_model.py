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
import sys
import time


def main():
    
    #load halo catalogue
    start = time.time()
    from halotools import sim_manager
    halocat = sim_manager.CachedHaloCatalog()
    #use custom processed Bolshoi halo catalogue
    """
    halocat = sim_manager.CachedHaloCatalog(simname = 'Bolshoi_250',
                                            redshift=0.0,
                                            version_name='1.0',
                                            halo_finder='Rockstar')
    """
    halo_table = halocat.halo_table
    print("time to load halo catalogue: ", time.time()-start)
    
    #define model
    start = time.time()
    from gal_halo_model import SMHM, BMHM
    sm_model = SMHM()
    sm_model.param_dict['scatter_model_param1'] = 0.2
    sm_model.param_dict['redshift'] = 0.0
    sm_model.param_dict['prim_haloprop_key'] = 'halo_mpeak'
    #sm_model.param_dict['smhm_beta_0']=0.6
    
    from halotools.empirical_models import SubhaloModelFactory
    model = SubhaloModelFactory(stellar_mass = sm_model)
    
    #get mock galaxy catalogue
    resolve_mock, resolve_a_mock, resolve_b_mock =\
        return_mock_resolve(halocat, model)
    print("time to build mock: ", time.time()-start)
    
    N = len(resolve_mock)
    print("number of galaxies in mock: ", N)
    
    #examine SMHM relation
    fig = plt.figure(figsize=(3.3,3.3))
    plt.subplots_adjust(left = 0.2, right = 0.9, top=0.9, bottom=0.2)
    plt.scatter(resolve_mock['halo_mvir'], resolve_mock['stellar_mass'], s=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$M_{*}~h^{-2}M_{\odot}$')
    plt.xlabel(r'$M_{\rm vir}~h^{-1}M_{\odot}$')
    plt.show()
    
    #examine stellar mass function
    bins = 10**np.arange(8.0,12.0,0.2)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    counts = np.histogram(resolve_mock['stellar_mass'],bins=bins)[0]
    err = np.sqrt(counts)
    
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.errorbar(bin_centers,counts,yerr=err, fmt='.')
    plt.ylim([1,10**3])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{*} ~M_{\odot}$')
    plt.ylabel(r'$N$')
    plt.show()
    
    #find groups
    start = time.time()
    from resolve_grp_dyn.group_finder import FoFGroups
    coords = np.vstack((resolve_mock['observed_x'],
                        resolve_mock['observed_y'],
                        resolve_mock['observed_z'])).T
    groups = FoFGroups(coords, 1.0, 1.0)
    group_ids = groups.group_ids
    print("time to find groups: ", time.time()-start)
    
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.scatter(resolve_mock['observed_x'],resolve_mock['observed_y'],c=group_ids, lw=0)
    plt.show()


def return_mock_resolve(halocat, model, gal_prop_cut=10**8.0):
    """
    return RESOLVE mock galaxy catalogue
    """
    
    #define cosmology for mock
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    
    #populate mock
    sub_start = time.time()
    model.populate_mock(halocat = halocat)
    print("time to get observed positions: ", time.time()-sub_start)
    
    #define galaxy sample
    if model.model_dictionary.has_key('stellar_mass'): gal_prop='stellar_mass'
    elif model.model_dictionary.has_key('baryonicr_mass'): gal_prop='baryonic_mass'
    mask = model.mock.galaxy_table[gal_prop] > gal_prop_cut
    galaxy_sample = model.mock.galaxy_table[mask]
    
    #place observer at center of box
    sub_start = time.time()
    galaxy_sample['halo_x'] = galaxy_sample['halo_x'] - halocat.Lbox/2.0
    galaxy_sample['halo_y'] = galaxy_sample['halo_y'] - halocat.Lbox/2.0
    galaxy_sample['halo_z'] = galaxy_sample['halo_z'] - halocat.Lbox/2.0
    pos = np.vstack((galaxy_sample['halo_x'],galaxy_sample['halo_y'],galaxy_sample['halo_z'])).T
    vels = coords = np.vstack((galaxy_sample['halo_vx'],galaxy_sample['halo_vy'],galaxy_sample['halo_vz'])).T
    
    from halotools.mock_observables import ra_dec_z
    ra,dec,redshift = ra_dec_z(pos, vels, cosmo=cosmo)
    ra = np.degrees(ra)
    dec= np.degrees(dec)
    galaxy_sample['ra'] = ra
    galaxy_sample['dec'] = dec
    galaxy_sample['redshift'] = redshift
    print("time to get ra,dec,z positions: ", time.time()-sub_start)
    
    #make ra,dec cuts
    resolve_mask, resolve_a_mask, resolve_b_mask = apply_resolve_mask(ra,dec,redshift)
    
    #calculate observed positions for galaxies in survey
    sub_start = time.time()
    r = cosmo.comoving_distance(redshift[resolve_mask]) #radial position
    theta = np.radians(np.fabs(dec[resolve_mask]-90.0)) #polar angle
    phi = np.radians(ra[resolve_mask] + 180.0) #azimuthal angle
    galaxy_sample['observed_x'] = 0.0
    galaxy_sample['observed_y'] = 0.0
    galaxy_sample['observed_z'] = 0.0
    galaxy_sample['observed_x'][resolve_mask] = r*np.sin(theta)*np.cos(phi)
    galaxy_sample['observed_y'][resolve_mask] = r*np.sin(theta)*np.sin(phi)
    galaxy_sample['observed_z'][resolve_mask] = r*np.cos(theta)
    print("time to get observed positions: ", time.time()-sub_start)
    
    return galaxy_sample[resolve_mask],\
           galaxy_sample[resolve_a_mask],\
           galaxy_sample[resolve_b_mask]

def apply_resolve_mask(ra,dec,redshift):
    """
    apply resolve ra,dec,and redshifts cuts
    
    Parameters
    ----------
    ra : array_like
        right acension of galaxies in degrees (-180,180)
        
    dec : array_like
        declination of galaxies in degrees (-90,90)
        
    redshift : array_like
        redshift of galaxies
    
    Returns
    -------
    resolve_mask, resolve_a_mask, resolve_b_mask
    """
    
    from astropy.constants import c
    speed_of_light = c.to('km/s').value
    
    resolve_b = {'min_ra': (3.0*(360.0/24.0)-180.0),
                 'max_ra': (22.0*(360.0/24.0)-180.0),
                 'min_dec': -1.25,
                 'max_dec': 1.25,
                 'min_redshift': 4500.0/speed_of_light,
                 'max_redshift': 7000.0/speed_of_light,
                 'wrap_ra' : True
                 }
    
    resolve_a = {'min_ra': (236.25-180.0),
                 'max_ra': (131.25-180.0),
                 'min_dec': 0.0,
                 'max_dec': 5.0,
                 'min_redshift': 4500.0/speed_of_light,
                 'max_redshift': 7000.0/speed_of_light,
                 'wrap_ra' : False
                 }
    
    #make ra-dec cut
    resolve_a_radec_mask = ra_dec_box_mask(ra,dec,resolve_a)
    resolve_b_radec_mask = ra_dec_box_mask(ra,dec,resolve_b)
    
    #make redshift cut
    resolve_a_z_mask = (redshift > resolve_a['min_redshift']) & (redshift < resolve_a['max_redshift'])
    resolve_b_z_mask = (redshift > resolve_b['min_redshift']) & (redshift < resolve_b['max_redshift'])
    
    #make combined mask
    resolve_a_mask = resolve_a_radec_mask & resolve_a_z_mask
    resolve_b_mask = resolve_b_radec_mask & resolve_b_z_mask
    resolve_mask = resolve_b_mask | resolve_a_mask
    
    return resolve_mask, resolve_a_mask, resolve_b_mask

def ra_dec_box_mask(ra,dec,params):
    """
    return boolean mask indicating whether points are inside a ra-dec box defined by 
    params dictionary
    
    Parameters
    ----------
    ra : array_like
        right acension of galaxies in degrees (-180,180)
        
    dec : array_like
        declination of galaxies in degrees (-90,90)
    
    params : dictionary
        dictionary indicating ra-dec bounda of box
    
    Returns
    --------
    mask
    """
    
    if params['wrap_ra']:
        ra_mask = (params['max_ra'] < ra) | (ra < params['min_ra'])
    else:
        ra_mask = (params['max_ra'] < ra) & (ra < params['min_ra'])
    
    dec_mask = (params['min_dec'] < dec) & (dec < params['max_dec'])
    
    mask = ra_mask & dec_mask
    
    return mask
    
    
def plot_sky_dist(ra,dec):
    """
    plt the positions of galaxies in ra,dec
    """
    
    #compress along dec 
    y = np.cos(np.radians(dec)-np.pi/2.0)
    x = ra
    
    #create plot
    new_tick_locations = np.array([-90.0,-60.0,-30.0,0.0,60.0,30.0,90.0])
    new_ticks = np.array([r'$-90^{\circ}$',r'$-60^{\circ}$',r'$-30^{\circ}$',r'$0^{\circ}$',r'$60^{\circ}$',r'$30^{\circ}$',r'$90^{\circ}$'])
    def tick_function(X):
        V = np.cos(np.radians(X)-np.pi/2.0)
        return [z for z in V]
    
    fig = plt.figure(figsize=(3.3,3.3))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left = 0.2, right = 0.9, top=0.9, bottom=0.2)
    #ax.grid(True)
    ax.scatter(x, y, s=1, alpha=1, lw=0)
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
