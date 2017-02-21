"""
    biclustlib: A Python library of biclustering algorithms and evaluation measures.
    Copyright (C) 2017  Victor Alexandre Padilha

    This file is part of biclustlib.

    biclustlib is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    biclustlib is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from ._base import BaseBiclusteringAlgorithm
from sklearn.cluster.bicluster import SpectralBiclustering
from ..model import Bicluster, Biclustering

class Spectral(BaseBiclusteringAlgorithm):
    """Spectral Biclustering. This class is a simple wrapper of Spectral's implementation available in
    the scikit-learn package.

    Spectral is an algorithm that uses the singular value decomposition to find biclusterings with
    checkerboard structures.

    Reference
    ---------
    Kluger, Y., Basri, R., Chang, J. T., and Gerstein, M. (2003). Spectral biclustering of microarray data:
    coclustering genes and conditions. Genome research, 13(4):703-716.

    Parameters
    ----------
    num_clusters : int or tuple (num_row_clusters, num_col_clusters), default: 3
        Number of row and column clusters. If integer, the algorithm will search for the same number
        of row and column clusters.

    normalization_method : str, default: 'bistochastic'
        Method of normalization. One between 'log', 'scale' or 'bistochastic'.

    num_vectors : int, default: 6
        Number of eigenvectors to check for the checkerboard structure.

    num_best : int, default: 2
        Number of eigenvectors to which the data will be projected before the clustering step.

    num_jobs : int, default: 1
        Number of parallel jobs to perform the algorithm.

    **params: dict
        See sklearn.cluster.bicluster.SpectralBiclustering documentation.
    """

    def __init__(self, num_clusters=3, normalization_method='bistochastic', num_vectors=6, num_best=2, num_jobs=1, **params):
        self.num_clusters = num_clusters
        self.spectral = SpectralBiclustering(num_clusters, normalization_method, num_vectors, num_best, n_jobs=num_jobs, **params)

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        self.sb.fit(data)

        try:
            num_row_clusters, num_col_clusters = self.num_clusters
        except TypeError:
            num_row_clusters = num_col_clusters = self.num_clusters

        num_biclusters = num_row_clusters * num_col_clusters

        return Biclustering([Bicluster(*self.sb.get_indices(i)) for i in range(num_biclusters)])
