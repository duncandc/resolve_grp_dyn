#!/usr/bin/python

#Author: Duncan Campbell
#June, 2016

###packages###
from __future__ import ( division, print_function, absolute_import, unicode_literals)
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
    
    gal_table = Table.read(filepath+filename, format='ascii.csv')
    
    #for name in gal_table.colnames:
    #    print(name)
    
    #get galaxies in the RESOLVE volume
    from galaxy_properties import resolve_volume
    resolve_a, resolve_b, a_buffer, b_buffer = resolve_volume(gal_table['radeg'], gal_table['dedeg'], gal_table['grpcz'])
    resolve = (resolve_a | resolve_b)
    
    #only keep galaxies in groups within the main resolve volumes
    gal_table =  gal_table[resolve]
    
    #define red/blue
    from galaxy_properties import red_blue_desgination
    red, blue = red_blue_desgination(gal_table['rmag'], gal_table['modelg_r'])
    
    #calculate baryonic mass
    from galaxy_properties import baryonic_mass
    gal_table['mbary'] = baryonic_mass(gal_table['logmstar'],gal_table['mhi'])
    gal_table['mstar'] = 10.0**gal_table['logmstar']
    
    #define mass bins
    bins = 10**np.arange(8.0,12.0,0.2)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    #baryonic mass
    counts_all = np.histogram(gal_table['mbary'],bins=bins)[0]
    counts_red = np.histogram(gal_table['mbary'][red],bins=bins)[0]
    counts_blue = np.histogram(gal_table['mbary'][blue],bins=bins)[0]
    err_all = np.sqrt(counts_all)
    err_red = np.sqrt(counts_red)
    err_blue = np.sqrt(counts_blue)
    
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.errorbar(bin_centers,counts_all,yerr=err_all, fmt='.', color='black')
    plt.errorbar(bin_centers,counts_red,yerr=err_all, fmt='.', color='red')
    plt.errorbar(bin_centers,counts_blue,yerr=err_all, fmt='.', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{\rm bary} ~M_{\odot}$')
    plt.ylabel(r'$N$')
    plt.show()
    
    filename = 'resolve_mbary_mass_func.dat'
    t = Table()
    t['mbary'] = bin_centers
    t['all'] = counts_all
    t['red'] = counts_red
    t['blue'] = counts_blue
    t.write(savepath+filename, format='ascii')
    
    counts_all = np.histogram(gal_table['mstar'],bins=bins)[0]
    counts_red = np.histogram(gal_table['mstar'][red],bins=bins)[0]
    counts_blue = np.histogram(gal_table['mstar'][blue],bins=bins)[0]
    err_all = np.sqrt(counts_all)
    err_red = np.sqrt(counts_red)
    err_blue = np.sqrt(counts_blue)
    
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.errorbar(bin_centers,counts_all,yerr=err_all, fmt='.', color='black')
    plt.errorbar(bin_centers,counts_red,yerr=err_all, fmt='.', color='red')
    plt.errorbar(bin_centers,counts_blue,yerr=err_all, fmt='.', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{*} ~M_{\odot}$')
    plt.ylabel(r'$N$')
    plt.show()
    
    filename = 'resolve_mstar_mass_func.dat'
    t = Table()
    t['mstar'] = bin_centers
    t['all'] = counts_all
    t['red'] = counts_red
    t['blue'] = counts_blue
    t.write(savepath+filename, format='ascii')

if __name__ == '__main__':
    main()