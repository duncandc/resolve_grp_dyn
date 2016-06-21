#!/usr/bin/python

#Author: Duncan Campbell
#June 12, 2016

###packages###
import numpy as np
import h5py
from astropy.table import Table
from scipy.stats import rankdata
from halotools.utils import group_member_generator
import matplotlib.pyplot as plt
import sys

def main():
    
    filepath = '../data/'
    savepath = '../output/'
    
    filename = 'resolve.csv'
    
    galaxy_mass_key = 'mbary'
    #galaxy_mass_key = 'mstar'
    
    gal_table = Table.read(filepath+filename, format='ascii.csv')
    
    #get galaxies in the RESOLVE volume
    from galaxy_properties import resolve_volume
    resolve_a, resolve_b, a_buffer, b_buffer = resolve_volume(gal_table['radeg'],gal_table['dedeg'],gal_table['grpcz'])
    resolve = (resolve_a | resolve_b)
    gal_table = gal_table[resolve]
    
    #add baryon mass column
    from galaxy_properties import baryonic_mass
    gal_table['mbary'] = baryonic_mass(gal_table['logmstar'],gal_table['mhi'])
    gal_table['mstar'] = 10.0**gal_table['logmstar']
    gal_table['logmbary'] = np.log10(gal_table['mbary'])
    
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
    gal_table = group_ranking(gal_table, grouping_key='grp', ranking_key=galaxy_mass_key, inverse_rank=True)
    
    #specify central satellite designation
    from group_properties import broadcast_central_prop
    if galaxy_mass_key=='mbary':
        cen = (gal_table['logmbary_rank']==1)
        sat = (gal_table['logmbary_rank']>1)
        gal_table['cen'] = 1
        gal_table['cen'][sat] = 0
        gal_table = broadcast_central_prop(gal_table, broadcast_key='logmbary',
                                       grouping_key='grp', central_key='cen')
    elif galaxy_mass_key=='mstar':
        cen = (gal_table['logmstar_rank']==1)
        sat = (gal_table['logmstar_rank']>1)
        gal_table['cen'] = 1
        gal_table['cen'][sat] = 0
        gal_table = broadcast_central_prop(gal_table, broadcast_key='logmstar',
                                       grouping_key='grp', central_key='cen')
    
    from group_properties import los_sat_delta_v
    gal_table = los_sat_delta_v(gal_table, grouping_key='grp', central_key='cen', velocity_key='cz')
    
    #calculate the velocity dispersion as a function of central mass
    bins = np.arange(8.5,12.0,0.25)
    bin_centers = (bins[1:]+bins[:-1])/2.0
    
    if galaxy_mass_key=='mbary':
        bin_key = 'logmstar_cen_mstar'
        value_key = 'd_cz_cen_mstar'
    elif galaxy_mass_key=='mstar':
        bin_key = 'logmbary_cen_mbary'
        value_key = 'd_cz_cen_mbary'
    stds = binned_std(gal_table[sat], bins, bin_key, value_key)[1]
    
    #calculate bootstrap errors
    N_bootstraps = 500
    unq_group_ids = np.unique(gal_table['grp'])
    N_groups = len(unq_group_ids)
    stds_samples = np.zeros((N_bootstraps, len(bin_centers)))
    for i in range(0,N_bootstraps):
        print(i)
        group_ids_to_keep = np.random.choice(unq_group_ids, size=N_groups)
        mask = np.in1d(gal_table['grp'], group_ids_to_keep)
        stds_samples[i,:] = binned_std(gal_table[sat & mask], bins, bin_key, value_key)[1]
    
    std_err = np.std(stds_samples, axis=0)
    
    #plot result
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.plot([5,15],[0,0], '--')
    plt.scatter(10**gal_table[bin_key][sat], np.fabs(gal_table[value_key][sat]),
                c='black', lw=0, s=5)
    plt.errorbar(10**bin_centers, stds_2, xerr=np.diff(10**bins/2.0), yerr=std_2_err,
                 color='magenta', fmt='o', mec='none')
    plt.ylim([10,1000])
    plt.xlim([10**8.0,10**12.0])
    plt.xscale('log')
    plt.ylabel(r'$\Delta v_{\rm LOS} ~{\rm km/s}$')
    if galaxy_mass_key=='mbary': plt.xlabel(r'$M_{\rm bary, cen} ~ M_{\odot}$')
    if galaxy_mass_key=='mstar': plt.xlabel(r'$M_{*, \rm cen} ~ M_{\odot}$')
    plt.show()
    
    fig.savefig('/Users/duncan/Desktop/resolve_sat_kin_mbary.pdf', dpi=300)
    

def binned_std(x, bins, bin_key, value_key, use_log=False):

    inds = np.digitize(x[bin_key],bins=bins)
    result = np.empty(len(bins)-1)
    for i in range(0,len(bins)-1):
        in_bin = (inds== i+1)
        if use_log==True:
            result[i] = np.std(np.log10(x[value_key][in_bin]))
        else:
            result[i] = np.std(x[value_key][in_bin])
        
    return bins, result

if __name__ == '__main__':
    main()