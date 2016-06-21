
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
from astropy.io import fits

def main():
    
    filepath = './data/'
    savepath = './output/'
    
    filename = 'resolvecatalog_str.fits'
    
    
    hdulist = fits.open(filepath+filename)
    
    print(hdulist[1].header)
    
    gal_table_1 = Table(hdulist[1].data)
    print(gal_table_1)
    
    gal_table_2 = Table(hdulist[2].data)
    print(gal_table_2)
    
    print(gal_table_1==gal_table_2)

if __name__ == '__main__':
    main()