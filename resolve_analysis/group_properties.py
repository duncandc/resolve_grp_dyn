
###packages###
from __future__ import ( division, print_function, absolute_import, unicode_literals)
import numpy as np
from astropy.table import Table
from scipy.stats import rankdata
from halotools.utils import group_member_generator

__all__=['group_ranking', 'group_multiplicity', 'los_sat_delta_v',
         'broadcast_central_prop']
__author__=['Duncan Campbell']

def group_ranking(gal_table, grouping_key='group_id', ranking_key='stellar_mass',
                       inverse_rank=True, assume_sorted=False, rank_method='ordinal'):
    """
    within groups specified by ``grouping_key``, calculate the rank of members 
    on ``ranking_key``.
    
    Parameters
    ==========
    gal_table : astropy.table object
    
    grouping_key : string
        string indicating grouping IDs
    
    ranking_key : string
    
    rank_method : string, optional
    
    inverse_rank : boolean, optional
    
    assume_sorted : boolean, optional
    
    Returns
    =======
    new_gal_table : astropy.table object
    """
    
    #sort by group ID
    if assume_sorted==False:
        sort_inds = np.argsort(np.array(gal_table), order=[grouping_key])
        gal_table = gal_table[sort_inds]
    
    #identify groups
    requested_columns = [ranking_key]
    group_gen = group_member_generator(gal_table, grouping_key, requested_columns)
    
    result = np.zeros(len(gal_table))
    if inverse_rank: a=-1.0 #used to reverse ranking prop
    else: a=1.0
    for first, last, member_props in group_gen:
        masses = a*member_props[0]
        rank = rankdata(masses, method=rank_method)
        result[first:last] = rank
    
    new_column_name = ranking_key + '_rank'
    gal_table[new_column_name] = result
    
    #resort by rank within groups
    sort_inds = np.argsort(np.array(gal_table), order=(grouping_key, ranking_key))
    gal_table = gal_table[sort_inds]
    
    return gal_table


def group_multiplicity(gal_table, grouping_key='group_id', assume_sorted=False):
    """
    calculate the number of group members specified by ``grouping_key``.
    
    Parameters
    ==========
    gal_table : astropy.table object
    
    grouping_key : string
        string indicating grouping IDs
    
    assume_sorted : boolean, optional
    
    Returns
    =======
    new_gal_table : astropy.table object
    """
    
    #sort by group ID
    if assume_sorted==False:
        sort_inds = np.argsort(np.array(gal_table), order=[grouping_key])
        gal_table = gal_table[sort_inds]
    
    #identify groups
    requested_columns = [grouping_key]
    group_gen = group_member_generator(gal_table, grouping_key, requested_columns)
    
    result = np.zeros(len(gal_table))
    for first, last, member_props in group_gen:
        N = len(member_props[0])
        result[first:last] = N
    
    new_column_name = grouping_key + '_multiplicity'
    gal_table[new_column_name] = result
    
    return gal_table
    

def los_sat_delta_v(gal_table, grouping_key='group_id', central_key='cen',
                    velocity_key='cz', assume_sorted=False):
    """
    calculate the LOS difference in velocoty between central and satellites in groups
    
    Parameters
    ==========
    gal_table : astropy.table object
    
    grouping_key : string
        string indicating grouping IDs
    
    central_key : string
        1 if central, 0 if satellite
    
    velocity_key : string
        LOS velocity
    
    assume_sorted : boolean, optional
    
    Returns
    =======
    new_gal_table : astropy.table object
    """
    
    #sort by group ID
    if assume_sorted==False:
        sort_inds = np.argsort(np.array(gal_table), order=(grouping_key, central_key))
        gal_table = gal_table[sort_inds]
    
    #identify groups
    requested_columns = [velocity_key]
    group_gen = group_member_generator(gal_table, grouping_key, requested_columns)
    
    result = np.zeros(len(gal_table))
    for first, last, member_props in group_gen:
        cen_v = member_props[0][-1]
        dv = member_props[0] - cen_v
        result[first:last] = dv
    
    new_column_name = 'd_'+velocity_key+'_'+central_key
    gal_table[new_column_name] = result
    
    return gal_table


def broadcast_central_prop(gal_table, broadcast_key,
                           grouping_key='group_id', central_key='cen',
                           assume_sorted=False):
    """
    calculate the LOS difference in velocoty between central and satellites in groups
    
    Parameters
    ==========
    gal_table : astropy.table object
    
    broadcast_key : string
    
    grouping_key : string
        string indicating grouping IDs
    
    central_key : string
        1 if central, 0 if satellite
    
    velocity_key : string
        LOS velocity
    
    assume_sorted : boolean, optional
    
    Returns
    =======
    new_gal_table : astropy.table object
    """
    
    #sort by group ID
    if assume_sorted==False:
        sort_inds = np.argsort(np.array(gal_table), order=(grouping_key, central_key))
        gal_table = gal_table[sort_inds]
    
    #identify groups
    requested_columns = [broadcast_key]
    group_gen = group_member_generator(gal_table, grouping_key, requested_columns)
    
    result = np.zeros(len(gal_table))
    for first, last, member_props in group_gen:
        cen_prop = member_props[0][-1]
        result[first:last] = cen_prop
    
    new_column_name = broadcast_key + '_' + central_key
    gal_table[new_column_name] = result
    
    return gal_table
