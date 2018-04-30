#!/usr/bin/python

# Author: Duncan Campbell
# June 12, 2016

import numpy as np
import h5py
from astropy.table import Table
from scipy.stats import rankdata
import halotools
from halotools.utils import group_member_generator
import matplotlib.pyplot as plt


def main():

    # load resolve data
    filepath = '../data/'
    filename = 'resolve.csv'
    gal_table = Table.read(filepath+filename, format='ascii.csv')

    gal_table['baryon_mass'] = 10.0**gal_table['logmstar'] + gal_table['mhi']
    gal_table['stellar_mass'] = 10.0**gal_table['logmstar']

    # define model
    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('behroozi10')
    model.param_dict['scatter_model_param1'] = 0.2

    # load halo catalogue
    from halotools import sim_manager
    halocat = sim_manager.CachedHaloCatalog(simname='bolshoi',
                                            redshift=0.0,
                                            version_name='halotools_v0p4',
                                            halo_finder='rockstar')

    # build mock
    model.populate_mock(halocat=halocat)
    print(model.mock.galaxy_table)

    # define galaxy sample
    mask = model.mock.galaxy_table['stellar_mass'] > 10**9.0
    galaxy_sample = model.mock.galaxy_table[mask]

    # assign extra galaxy properties
    sort_inds = np.argsort(gal_table['stellar_mass'])
    gal_table = gal_table[sort_inds]
    inds = np.searchsorted(gal_table['stellar_mass'], galaxy_sample['stellar_mass'])
    mask = (inds == len(gal_table))
    inds[mask] = len(gal_table)-1
    galaxy_sample['baryon_mass'] = gal_table['baryon_mass'][inds]

    # get redshift space positions
    x = galaxy_sample['x']
    y = galaxy_sample['y']
    z = galaxy_sample['z']
    vz = galaxy_sample['vz']
    from halotools.mock_observables import return_xyz_formatted_array
    pos = return_xyz_formatted_array(x, y, z, velocity=vz, velocity_distortion_dimension='z')

    # find groups
    from halotools.mock_observables import FoFGroups

    b_perp, b_para = (0.07, 1.1)
    groups = FoFGroups(pos, b_perp, b_para, period=halocat.Lbox)
    galaxy_sample['group_id'] = groups.group_ids

    # sort by group ID
    sort_inds = np.argsort(np.array(galaxy_sample), order=['group_id'])
    galaxy_sample = galaxy_sample[sort_inds]

    # find group ranks
    grouping_key = 'group_id'
    requested_columns = ['stellar_mass']
    group_gen = group_member_generator(galaxy_sample, grouping_key, requested_columns)

    result = np.zeros(len(galaxy_sample))
    for first, last, member_props in group_gen:
        masses = -1.0*member_props[0]
        rank = rankdata(masses, method='ordinal')
        result[first:last] = rank
    galaxy_sample['rank'] = result

    # resort by rank within groups
    sort_inds = np.argsort(np.array(galaxy_sample), order=('group_id', 'rank'))
    galaxy_sample = galaxy_sample[sort_inds]

    # get central props
    grouping_key = 'group_id'
    requested_columns = ['stellar_mass', 'baryon_mass']
    group_gen = group_member_generator(galaxy_sample, grouping_key, requested_columns)

    result1 = np.zeros(len(galaxy_sample))
    result2 = np.zeros(len(galaxy_sample))
    for first, last, member_props in group_gen:
        result1[first:last] = member_props[0][0]
        result2[first:last] = member_props[1][0]
    galaxy_sample['stellar_mass_cen'] = result1
    galaxy_sample['baryon_mass_cen'] = result2

    # find group centric velocities
    x = galaxy_sample['x']
    y = galaxy_sample['y']
    z = galaxy_sample['z']
    vx = galaxy_sample['vx']
    vy = galaxy_sample['vy']
    vz = galaxy_sample['vz']
    pos = np.vstack((x, y, z)).T
    vels = np.vstack((vx, vy, vz)).T

    from astropy.cosmology import FlatLambdaCDM
    from scipy.interpolate import interp1d
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    yy = np.arange(0, 1.0, 0.001)
    xx = cosmo.comoving_distance(yy).value
    f = interp1d(xx, yy, kind='cubic')
    from astropy.constants import c
    c_km_s = c.to('km/s').value
    z_cos = f(z)
    redshift = z_cos+(vz/c_km_s)*(1.0+z_cos)
    galaxy_sample['cz'] = redshift*c_km_s

    # get group centric velocities
    grouping_key = 'group_id'
    requested_columns = ['cz']
    group_gen = group_member_generator(galaxy_sample, grouping_key, requested_columns)

    result = np.zeros(len(galaxy_sample))
    for first, last, member_props in group_gen:
        cen_cv = member_props[0][0]
        delta_cv = member_props[0] - cen_cv
        result[first:last] = delta_cv

    galaxy_sample['delta_cz'] = result

    print(galaxy_sample['delta_cz'])

    # get mask for cen/sat
    sat = (galaxy_sample['rank'] > 1)

    # calculate the std in mass bins
    bins = np.arange(8.5, 12.0, 0.2)
    bin_centers = (bins[1:]+bins[:-1])/2.0
    bin_key = 'stellar_mass_cen'
    value_key = 'delta_cz'
    stds = binned_std(galaxy_sample[sat], bins, bin_key, value_key)[1]

    fig = plt.figure(figsize=(3.3, 3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.plot([10**5, 10**15], [0, 0], '--')
    plt.scatter(galaxy_sample['stellar_mass_cen'][sat], galaxy_sample['delta_cz'][sat], lw=0, s=5, alpha=0.1, c='black')
    plt.plot(bin_centers, stds, color='black')
    plt.plot(bin_centers, -1.0*stds, color='black')
    plt.xscale('log')
    plt.ylim([-1000, 1000])
    plt.xlim([10**9.0, 10**12.0])
    plt.ylabel(r'$\Delta v_{\rm LOS} ~{\rm km/s}$')
    plt.xlabel(r'$M_{*} ~ M_{\odot}}$')
    plt.show()

    fig.savefig('/Users/duncan/Desktop/fig_3.pdf', dpi=300)

    fig = plt.figure(figsize=(3.3, 3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.plot([10**5, 10**15], [0, 0], '--')
    plt.scatter(galaxy_sample['baryon_mass_cen'][sat], galaxy_sample['delta_cz'][sat], lw=0, s=5, alpha=0.1, c='black')
    plt.plot(bin_centers, stds, color='black')
    plt.plot(bin_centers, -1.0*stds, color='black')
    plt.xscale('log')
    plt.ylim([-1000, 1000])
    plt.xlim([10**9.0, 10**12.0])
    plt.ylabel(r'$\Delta v_{\rm LOS} ~{\rm km/s}$')
    plt.xlabel(r'$M_{\rm baryon} ~ M_{\odot}}$')
    plt.show()

    fig.savefig('/Users/duncan/Desktop/fig_4.pdf', dpi=300)


def binned_std(x, bins, bin_key, value_key, use_log=False):

    inds = np.digitize(x[bin_key], bins=bins)
    result = np.empty(len(bins)-1)
    for i in range(0, len(bins)-1):
        in_bin = (inds == i+1)
        if use_log == True:
            result[i] = np.std(np.log10(x[value_key][in_bin]))
        else:
            result[i] = np.std(x[value_key][in_bin])

    return bins, result


if __name__ == '__main__':
    main()
