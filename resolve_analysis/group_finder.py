"""
Module containing the `~halotools.mock_observables.FoFGroups` class used to
identify friends-of-friends groups of points.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse import csgraph, csr_matrix, coo_matrix

from pair_counter import pairwise_distance_rp_pi

from halotools.custom_exceptions import HalotoolsError

igraph_available=True
try:
    import igraph
except ImportError:
    igraph_available=False
if igraph_available is True:  # there is another package called igraph--need to distinguish.
    if not hasattr(igraph, 'Graph'):
        igraph_available is False
no_igraph_msg = ("igraph package not installed.  Some functions will not be available. \n"
    "See http://igraph.org/ and note that there are two packages called 'igraph'.")

__all__ = ['FoFGroups']
__author__ = ['Duncan Campbell']


class FoFGroups(object):
    """
    Friends-of-friends (FoF) groups class.
    """

    def __init__(self, positions, d_perp, d_para):
        """
        Build FoF groups in redshift space assuming an observer at (0,0,0).

        Parameters
        ----------
        positions : array_like
            Npts x 3 numpy array containing 3-D positions of galaxies.
            Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        d_perp : float
            Maximum linking length in the perpendicular direction.

        d_para : float
            Maximum linking length in the parallel direction.

        """

        self.d_perp = float(d_perp)  # perpendicular linking length
        self.d_para = float(d_para)  # parallel linking length
        self.positions=np.asarray(positions, dtype=np.float64)  # coordinates of galaxies
        
        x = self.positions[:,0]
        y = self.positions[:,1]
        z = self.positions[:,2]
        
        rp, pi, i_inds, j_inds = pairwise_distance_rp_pi(x,y,z,x,y,z,self.d_perp,self.d_para)
        self.m_perp = coo_matrix((rp, (i_inds, j_inds)), shape=(len(x),len(x)))
        self.m_para = coo_matrix((pi, (i_inds, j_inds)), shape=(len(x),len(x)))

        self.m = self.m_perp.multiply(self.m_perp)+self.m_para.multiply(self.m_para)
        self.m = self.m.sqrt()

    @property
    def group_ids(self):
        """
        Determine integer IDs for groups.

        Each member of a group is assigned a unique integer ID that it shares with all
        connected group members.

        Returns
        -------
        group_ids : np.array
            array of group IDs for each galaxy

        """
        if getattr(self, '_group_ids', None) is None:
            self._n_groups, self._group_ids = csgraph.connected_components(
                self.m_perp, directed=False, return_labels=True)
        return self._group_ids

    @property
    def n_groups(self):
        """
        Calculate the total number of groups, including 1-member groups

        Returns
        -------
        N_groups: int
            number of distinct groups

        """
        if getattr(self, '_n_groups', None) is None:
            self._n_groups = csgraph.connected_components(self.m_perp,
                directed=False, return_labels=False)
        return self._n_groups

    def create_graph(self):
        """
        Create graph from FoF sparse matrix (requires igraph package).
        """
        if igraph_available is True:
            self.g = _scipy_to_igraph(self.m, self.positions, directed=False)
        else:
            raise HalotoolsError(no_igraph_msg)


    def get_degree(self):
        """
        Calculate the 'degree' of each galaxy vertex (requires igraph package).

        Returns
        -------
        degree : np.array
            the 'degree' of galaxies in groups

        """
        if igraph_available is True:
            self.degree = self.g.degree()
            return self.degree
        else:
            raise HalotoolsError(no_igraph_msg)


    def get_betweenness(self):
        """
        Calculate the 'betweenness' of each galaxy vertex (requires igraph package).

        Returns
        -------
        betweeness : np.array
            the 'betweenness' of galaxies in groups
        """
        if igraph_available is True:
            self.betweenness = self.g.betweenness()
            return self.betweenness
        else:
            raise HalotoolsError(no_igraph_msg)


    def get_multiplicity(self):
        """
        Return the multiplicity of galaxies' group (requires igraph package).
        """
        if igraph_available is True:
            clusters = self.g.clusters()
            mltp = np.array(clusters.sizes())
            self.multiplicity = mltp[self.group_ids]
            return self.multiplicity
        else:
            raise HalotoolsError(no_igraph_msg)


    def get_edges(self):
        """
        Return all edges of the graph (requires igraph package).

        Returns
        -------
        edges: np.ndarray
            N_edges x 2 array of vertices that are connected by an edge.  The vertices are
            indicated by their index.

        """
        if igraph_available is True:
            self.edges = np.asarray(self.g.get_edgelist())
            return self.edges
        else: raise HalotoolsError(no_igraph_msg)


    def get_edge_lengths(self):
        """
        Return the length of all edges (requires igraph package).

        Returns
        -------
        lengths: np.array
            The length of an 'edge' econnnecting galaxies, i.e. distance between galaxies.

        Notes
        ------
        The length is caclulated as:

        .. math::
            L_{\\rm edge} = \\sqrt{r_{\\perp}^2 + r_{\\parallel}^2},

        where :math:`r_{\\perp}` and :math:`r_{\\parallel}` are the perendicular and
        parallel distance between galaixes.

        """
        if igraph_available is True:
            edges = self.g.es()
            lens = edges.get_attribute_values('weight')
            self.edge_lengths = np.array(lens)
            return self.edge_lengths
        else: raise HalotoolsError(no_igraph_msg)



def _scipy_to_igraph(matrix, coords, directed=False):
    """
    Convert a scipy sparse matrix to an igraph graph object (requires igraph package).

    Paramaters
    ----------
    matrix : object
        scipy.sparse pairwise distance matrix

    coords : np.array
        N by 3 array of coordinates of points

    Returns
    -------
    graph : object
        igraph graph object

    """

    matrix = csr_matrix(matrix)
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets].tolist()[0]

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    if igraph_available:
        g = igraph.Graph(list(zip(sources, targets)),
            n=matrix.shape[0], directed=directed,
            edge_attrs={'weight': weights},
            vertex_attrs={'x': x, 'y': y, 'z': z})
        return g
    else:
        raise HalotoolsError(no_igraph_msg)