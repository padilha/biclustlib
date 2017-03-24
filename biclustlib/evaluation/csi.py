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

import numpy as np

from scipy import sparse as sp

def csi(predicted_biclustering, reference_biclustering, num_rows, num_cols, sparse=True):
    """The Campello Soft Index (CSI) external evaluation measure.

    CSI computes the similarity between two soft clusterings. This measure was originally
    introduced in (Campello, 2010). In this package, it was implemented following the approach
    presented in (Horta and Campello, 2014), which first transforms each biclustering solution to
    a soft clustering representation and then applies the CSI measure. CSI lies in the interval
    [0, 1], where values close to 1 indicate better biclustering solutions.

    Reference
    ---------
    Campello, R. J. G. B. (2010). Generalized external indexes for comparing data partitions with
    overlapping categories. Pattern Recognition Letters, 31(9), 966-975.

    Horta, D., & Campello, R. J. G. B. (2014). Similarity measures for comparing biclusterings.
    IEEE/ACM Transactions on Computational Biology and Bioinformatics, 11(5), 942-954.

    Parameters
    ----------
    predicted_biclustering : biclustlib.model.Biclustering
        Predicted biclustering solution.

    reference_biclustering : biclustlib.model.Biclustering
        Reference biclustering solution.

    num_rows : int
        Number of rows of the dataset.

    num_cols : int
        Number of columns of the dataset.

    sparse : bool, default: True
        Wheter the (co)association matrices will be represented as sparse matrices. In most cases
        setting this parameter to True will increase computation efficiency.
    """
    predicted_clustering = _biclustering_to_soft_clustering(predicted_biclustering, num_rows, num_cols)
    reference_clustering = _biclustering_to_soft_clustering(reference_biclustering, num_rows, num_cols)

    predicted_association = _calculate_association(predicted_clustering, num_rows, num_cols, sparse)
    predicted_coassociation = _calculate_coassociation(predicted_association)

    reference_association = _calculate_association(reference_clustering, num_rows, num_cols, sparse)
    reference_coassociation = _calculate_coassociation(reference_association)

    agreements = _calculate_agreements(predicted_coassociation, reference_coassociation, predicted_association, reference_association, sparse)
    disagreements = _calculate_disagreements(predicted_coassociation, reference_coassociation, predicted_association, reference_association, sparse)

    return float(agreements) / (agreements + disagreements)

def _biclustering_to_soft_clustering(biclustering, num_rows, num_cols):
    is_singleton = np.ones(num_rows * num_cols, dtype=np.bool)
    soft_clustering = []

    for b in biclustering.biclusters:
        cluster = (b.rows[:, np.newaxis] + b.cols * num_rows).flatten()
        soft_clustering.append(cluster)
        is_singleton[cluster] = False

    soft_clustering.extend(i for i in np.where(is_singleton)[0])

    return soft_clustering

def _calculate_association(clustering, num_rows, num_cols, sparse):
    association = _zeros(clustering, num_rows, num_cols, sparse)

    for k, c in enumerate(clustering):
        association[k, c] = 1

    if sparse:
        return sp.csr_matrix(association)
    return association

def _calculate_coassociation(association):
    return association.T.dot(association)

def _calculate_agreements(ju, jv, u, v, sparse):
    return _triu(ju.minimum(jv), sparse).sum() + np.minimum(u.sum(axis=0) - 1, v.sum(axis=0) - 1).sum()

def _calculate_disagreements(ju, jv, u, v, sparse):
    return abs(_triu(ju - jv, sparse)).sum() + abs((u.sum(axis=0) - 1) - (v.sum(axis=0) - 1)).sum()

def _triu(a, sparse):
    if sparse:
        return sp.triu(a, k=1).sum()
    return np.triu(a, k=1).sum()

def _zeros(clustering, num_rows, num_cols, sparse):
    if sparse:
        return sp.dok_matrix((len(clustering), num_rows * num_cols), dtype=np.int)
    return np.matrix(np.zeros((len(clustering), num_rows * num_cols), dtype=np.int))
