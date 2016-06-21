"""
Find groups on RESOLVE
"""

from __future__ import ( division, print_function, absolute_import, unicode_literals)
import numpy as np
from astropy.table import Table
from group_finder import FoFGroups
import sys
import matplotlib.pyplot as plt

def main():
    
    #load RESOLVE data
    filepath = '../data/'
    savepath = '../output/'
    filename = 'resolve.csv'
    gal_table = Table.read(filepath+filename, format='ascii.csv')
    
    #get baryonic masses
    from galaxy_properties import baryonic_mass
    gal_table['mbary'] = baryonic_mass(gal_table['logmstar'],gal_table['mhi'])
    
    #make completeness cut
    keep = (gal_table['mbary'] >= 10**9.3)
    gal_table = gal_table[keep]
    
    #count number of galaxies within primary volumes
    from galaxy_properties import resolve_volume
    resolve_a, resolve_b, a_buffer, b_buffer = resolve_volume(gal_table['radeg'],
                                                              gal_table['dedeg'],
                                                              gal_table['cz'])
    
    #number of galaxies used to calculate the number density and linking lengths
    N_gal = np.sum(resolve_a)+np.sum(resolve_b)
    print('number of galaxies intially in the volume: ', N_gal)
    
    #define cosmology
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    
    #find apparent positions in Mpc centered on observed at (0,0,0)
    x,y,z = get_cartesian_coordinates(gal_table['radeg'],
                                      gal_table['dedeg'],
                                      gal_table['cz'], cosmo)
    coords = np.vstack((x,y,z)).T
    gal_table['x']=x
    gal_table['y']=y
    gal_table['z']=z
    
    #find groups
    b_perp = 0.07
    b_para = 1.10
    volume_resolve_a = 38400.0*cosmo.h**3 #h^-3 Mpc^3
    volume_resolve_b = 13700.0*cosmo.h**3 #h^-3 Mpc^3
    volume_resolve = volume_resolve_a + volume_resolve_b
    print("volume of survey (h^-3 Mpc^3): ", volume_resolve)
    n_gal = N_gal/volume_resolve
    d_perp = b_perp/(n_gal**(1.0/3.0))
    d_para = b_para/(n_gal**(1.0/3.0))
    print("physical linking lenghts (h^-1 Mpc): ", d_perp, d_para)
    groups = FoFGroups(coords, d_perp, d_para)
    group_ids = groups.group_ids
    
    #replace group id's with our own
    gal_table['grp'] = group_ids
    
    #define central galaxies
    from group_properties import group_ranking
    gal_table = group_ranking(gal_table, grouping_key='grp', ranking_key='mbary', inverse_rank=True)
    cen = (gal_table['mbary_rank']==1)
    sat = (gal_table['mbary_rank']>1)
    gal_table['cen'] = 1
    gal_table['cen'][sat] = 0
    
    from group_properties import broadcast_central_prop
    gal_table = broadcast_central_prop(gal_table, broadcast_key='cz',
                                       grouping_key='grp', central_key='cen')
    
    #only keep galaxies in groups which fall within the primary volume 
    resolve_a, resolve_b, a_buffer, b_buffer = resolve_volume(gal_table['radeg'],
                                                              gal_table['dedeg'],
                                                              gal_table['cz_cen'])
    keep = (resolve_a | resolve_b)
    gal_table = gal_table[keep]
    N_gal = np.sum(resolve_a) + np.sum(resolve_b)
    print('number of galaxies in the volume w/ group corrections: ', N_gal)
    
    #make new table with complete sample and group IDs
    galaxy_id = np.arange(0,N_gal,1).astype('int')
    group_id = np.array(gal_table['grp']).astype('int')
    t = Table([galaxy_id, group_id], names=('id','group_id'))
    
    t['ra'] = gal_table['radeg']
    t['dec'] = gal_table['dedeg']
    t['cz'] = gal_table['cz']
    t['x'] = gal_table['x']
    t['y'] = gal_table['y']
    t['z'] = gal_table['z']
    t['central'] = gal_table['cen']
    
    filepath = '/Users/duncan/Desktop/'
    filename = 'resolve_bary_limited_groups.dat'
    t.write(filepath+filename, format='ascii')
    
    #how many groups are there?
    unq_group_ids = np.unique(t['group_id'])
    N_groups = len(unq_group_ids)
    print("total number of groups: ", N_groups)
    
    #calculate the multiplicity
    from group_properties import group_multiplicity
    t = group_multiplicity(t, grouping_key='group_id')
    
    mult_key = 'group_id_multiplicity'
    cen = (t['central']==1)
    bins = np.arange(0.5,100.5,1.0)
    N = (bins[:-1]+bins[1:])/2.0
    N_groups = np.histogram(t[mult_key][cen], bins=bins)[0]
    
    print(np.max(t[mult_key][cen]))
    
    #plot result
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    p1, = plt.plot(N,N_groups, color='black')
    plt.yscale('log')
    plt.xlabel(r'$N_{\rm members}$')
    plt.ylabel(r'$N_{\rm groups}$')
    plt.xlim([1,10])
    plt.ylim([0,1000])
    plt.show()
    
def get_cartesian_coordinates(ra, dec, cz, cosmo):
    """
    return cartesian coordinates assuming an observer at (0,0,0)
    """
    
    from astropy.constants import c
    speed_of_light = c.to('km/s').value
    redshift = np.array(cz)/speed_of_light
    
    r = cosmo.comoving_distance(redshift)*cosmo.h #radial distance h^-1 Mpc
    theta = np.radians(np.fabs(dec-90.0)) #polar angle
    phi = np.radians(ra) #azimuthal angle
    
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    
    return x,y,z

if __name__ == '__main__':
    main()