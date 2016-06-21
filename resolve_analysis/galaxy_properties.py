
"""
calculate derived galaxy properties in RESOLVE
"""

###packages###
from __future__ import ( division, print_function, absolute_import, unicode_literals)
import numpy as np

__all__=['red_blue_desgination', 'baryonic_mass', 'resolve_volume', 'completeness']
__author__=['Duncan Campbell']

def red_blue_desgination(Mr, g_r_color):
    """
    define 'red' and 'blue' galaxies using color cut
    
    Parameters
    ----------
    Mr : array_like
    
    g_r_color :  array_like
    
    Returns
    -------
    red : numpy.array
        boolean mask indicatiin red galaxies
    
    blue : numpy.array
        boolean mask indicatiin red galaxies
    """
    
    cut = 1.65 - 0.032*(Mr - 5.0*np.log10(0.7)+16.5)
    red = (g_r_color>cut)
    blue = (g_r_color<=cut)
    
    return red, blue


def baryonic_mass(mstar, mhi, log_mstar=True, log_mhi=False, return_log_mbary=False):
    """
    return total baryonic mass of galaxies
    """
    
    if log_mstar:
        mstar = 10.0**mstar
    if log_mhi:
        mhi = 10.0**mhi
    
    mbary = np.log10(mstar + mhi)
    
    if return_log_mbary:
        mbary
    else:
        return 10.0**mbary
    
def resolve_volume(ra, dec, cz):
    """
    return integer indicating if the galaxy falls in RESOLVE A, B, A buffer, B buffer
    
    Parameters
    ----------
    ra : array_like
        right acension
    
    dec : array_like
        declination
    
    cz : array_like
        recession velocity
    
    Returns
    -------
    resolve_a_mask : numpy.array
        boolean array indicating the galaxy falls in RESOLVE A volume
    
    resolve_b_mask : numpy.array
        boolean array indicating the galaxy falls in RESOLVE B volume
        
    resolve_a_buffer : numpy.array
        boolean array indicating the galaxy falls in RESOLVE A buffer region volume
        
    resolve_b_buffer : numpy.array
        boolean array indicating the galaxy falls in RESOLVE B buffer region volume
    """
    
    ra = ra-180.0
    
    resolve_b = {'min_ra': (3.0*(360.0/24.0)-180.0),
                 'max_ra': (22.0*(360.0/24.0)-180.0),
                 'min_dec': -1.25,
                 'max_dec': 1.25,
                 'min_redshift': 4500.0,
                 'max_redshift': 7000.0,
                 'wrap_ra' : True
                 }
    
    resolve_a = {'min_ra': (236.25-180.0),
                 'max_ra': (131.25-180.0),
                 'min_dec': 0.0,
                 'max_dec': 5.0,
                 'min_redshift': 4500.0,
                 'max_redshift': 7000.0,
                 'wrap_ra' : False
                 }
    
    #make ra-dec cut
    resolve_a_radec_mask = ra_dec_box_mask(ra,dec,resolve_a)
    resolve_b_radec_mask = ra_dec_box_mask(ra,dec,resolve_b)
    
    #make redshift cut
    resolve_a_z_mask = (cz > resolve_a['min_redshift']) & (cz < resolve_a['max_redshift'])
    resolve_b_z_mask = (cz > resolve_b['min_redshift']) & (cz < resolve_b['max_redshift'])
    
    #make combined mask
    resolve_a_mask = resolve_a_radec_mask & resolve_a_z_mask
    resolve_b_mask = resolve_b_radec_mask & resolve_b_z_mask
    
    resolve_a_buffer = (resolve_a_radec_mask == True) & (resolve_a_z_mask==False)
    resolve_b_buffer = (resolve_b_radec_mask == True) & (resolve_b_z_mask==False)
    
    return resolve_a_mask, resolve_b_mask, resolve_a_buffer, resolve_b_buffer


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


def completeness(gal_mass, volume='resolve_a', type='baryonic'):
    """
    return mask indictaing whether a galaxy falls in the completeness limit of the survey
    
    Parameters
    ----------
    gal_mass : array_like
        baryonic or stellar mass of galaxy
        
    volume : string, optional
        resolve_a, resolve_b, or resolve
    
    type : string, optional
        baryonic or stellar
    
    Returns
    -------
    mask : numpy.array
        boolean array indicating the galaxy passes the copleteness cut for the volume
    """
    
    gal_mass = np.log10(gal_mass)
    
    if volume=='resolve_a':
        if type=='baryonic':
            mask = (gal_mass >= 9.3)
        elif type=='stellar':
            mask = (gal_mass >= 9.0)
        else:
            print('must be type baryonic or stellar')
    elif volume=='resolve_b':
        if type=='baryonic':
            mask = (gal_mass >= 9.1)
        elif type=='stellar':
            mask = (gal_mass >= 9.0)
        else:
            print('must be type baryonic or stellar')
    elif volume=='resolve':
        if type=='baryonic':
            mask = (gal_mass >= 9.3)
        elif type=='stellar':
            mask = (gal_mass >= 9.0)
        else:
            print('must be type baryonic or stellar')
    else:
        print("volume must be resolve_a or resolve_b")
    
    return mask 

