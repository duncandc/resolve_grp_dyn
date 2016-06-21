#!/usr/bin/python

#Author: Duncan Campbell
#June 12, 2016

###packages###
from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
import h5py
from astropy.table import Table
from scipy.stats import rankdata
from halotools.utils import group_member_generator
import matplotlib.pyplot as plt
import sys

def main():
    
    galaxy_mass_key = 'mbary'
    #galaxy_mass_key = 'mstar'
    
    filepath = '../data/'
    savepath = '../output/'
    
    filename = 'resolve.csv'
    
    gal_table = Table.read(filepath+filename, format='ascii.csv')
    
    #remove low luminosity galaxies
    #keep = (gal_table['rmag'] <= 17.33)
    #gal_table = gal_table[keep]
    
    #calculate baryonic mass
    from galaxy_properties import baryonic_mass
    gal_table['mbary'] = baryonic_mass(gal_table['logmstar'],gal_table['mhi'])
    gal_table['mstar'] = 10.0**gal_table['logmstar']
    gal_table['logmbary'] = np.log10(gal_table['mbary'])
    
    #get galaxies in the RESOLVE volume
    from galaxy_properties import resolve_volume
    resolve_a, resolve_b, a_buffer, b_buffer = resolve_volume(gal_table['radeg'],gal_table['dedeg'],gal_table['grpcz'])
    resolve = (resolve_a | resolve_b)
    
    #split galaxy sampel into resolve a and b
    #gal_table_a = gal_table[resolve_a]
    #gal_table_b = gal_table[resolve_b]
    gal_table = gal_table[resolve]
    
    #make completeness cuts
    from galaxy_properties import completeness
    if galaxy_mass_key == 'mbary':
        keep = completeness(gal_table['mbary'], type='baryonic', volume='resolve')
    elif galaxy_mass_key == 'mstar':
        keep = completeness(gal_table['mstar'], type='stellar', volume='resolve')
    else:
        print("galaxy mass key not recognized.")
    gal_table=gal_table[keep]
    print("number of galaxies in sample: ", len(gal_table))
    
    #calculate ranks within groups
    from group_properties import group_ranking
    gal_table = group_ranking(gal_table, grouping_key='grp', ranking_key='logmstar', inverse_rank=True)
    gal_table = group_ranking(gal_table, grouping_key='grp', ranking_key='logmbary', inverse_rank=True)
    
    #calculate the multiplicity
    from group_properties import group_multiplicity
    gal_table = group_multiplicity(gal_table, grouping_key='grp')
    
    #define red/blue
    from galaxy_properties import red_blue_desgination
    red, blue = red_blue_desgination(gal_table['rmag'], gal_table['modelg_r'])
    
    #specify central satellite designation
    if galaxy_mass_key == 'mbary':
        cen = (gal_table['logmbary_rank']==1)
        sat = (gal_table['logmbary_rank']>1)
    elif galaxy_mass_key == 'mstar':
        cen = (gal_table['logmstar_rank']==1)
        sat = (gal_table['logmstar_rank']>1)
    
    #count the number of groups as a function of N_members
    mult_key = 'grp_multiplicity'
    
    bins = np.arange(0.5,100.5,1.0)
    N = (bins[:-1]+bins[1:])/2.0
    N_groups = np.histogram(gal_table[mult_key][cen], bins=bins)[0]
    N_red_groups = np.histogram(gal_table[mult_key][cen & red], bins=bins)[0]
    N_blue_groups = np.histogram(gal_table[mult_key][cen & blue], bins=bins)[0]
    
    #check for consistency
    print(np.all(N_groups == (N_red_groups+N_blue_groups)))
    
    #plot result
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    p1, = plt.plot(N,N_groups, color='black')
    p2, = plt.plot(N,N_red_groups, color='red')
    p3, = plt.plot(N,N_blue_groups, color='blue')
    plt.yscale('log')
    plt.xlabel(r'$N_{\rm members}$')
    plt.ylabel(r'$N_{\rm groups}$')
    plt.xlim([1,10])
    plt.ylim([0,1000])
    plt.legend((p1,p2,p3),('all', 'w/ red cen', 'w/ blue cen'), frameon=False, fontsize=10, title='FoF groups')
    plt.show()
    
    filename = 'resolve_'+galaxy_mass_key+'_group_n_func.dat'
    t = Table()
    t['N'] = N
    t['N_groups_all'] = N_groups
    t['N_groups_red'] = N_red_groups
    t['N_groups_blue'] = N_blue_groups
    t.write(savepath+filename, format='ascii')
    
if __name__ == '__main__':
    main()