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
from .check import check_biclusterings

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

    Returns
    -------
    csi_value : float
        Similarity score between 0.0 and 1.0.
    """

    check = check_biclusterings(predicted_biclustering, reference_biclustering)

    if isinstance(check, float):
        return check

    predicted_clustering = _biclustering_to_soft_clustering(predicted_biclustering, num_rows, num_cols)
    reference_clustering = _biclustering_to_soft_clustering(reference_biclustering, num_rows, num_cols)

    predicted_association = _calculate_association(predicted_clustering, num_rows, num_cols, sparse)
    predicted_coassociation = _calculate_coassociation(predicted_association)
    predicted_beta = _calculate_beta(predicted_association)

    reference_association = _calculate_association(reference_clustering, num_rows, num_cols, sparse)
    reference_coassociation = _calculate_coassociation(reference_association)
    reference_beta = _calculate_beta(reference_association)

    agreements = _calculate_agreements(predicted_coassociation, reference_coassociation, predicted_beta, reference_beta, sparse)
    disagreements = _calculate_disagreements(predicted_coassociation, reference_coassociation, predicted_beta, reference_beta, sparse)

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
    if sparse:
        association = sp.dok_matrix((len(clustering), num_rows * num_cols), dtype=np.int)
    else:
        association = np.zeros((len(clustering), num_rows * num_cols), dtype=np.int)

    for k, c in enumerate(clustering):
        association[k, c] = 1

    if sparse:
        return sp.csr_matrix(association)
    return association

def _calculate_coassociation(association):
    return association.T.dot(association)

def _calculate_beta(association):
    return association.sum(axis=0) - 1

def _calculate_agreements(predicted_coassociation, reference_coassociation, predicted_beta, reference_beta, sparse):
    num_objects = predicted_coassociation.shape[0]
    min_alpha = _triu(predicted_coassociation.minimum(reference_coassociation), sparse)
    min_beta = np.minimum(predicted_beta, reference_beta)
    return min_alpha.sum() + min_beta.sum() * (num_objects - 1)

def _calculate_disagreements(predicted_coassociation, reference_coassociation, predicted_beta, reference_beta, sparse):
    num_objects = predicted_coassociation.shape[0]
    abs_alpha = abs(_triu(predicted_coassociation - reference_coassociation, sparse))
    abs_beta = abs(predicted_beta - reference_beta)
    return abs_alpha.sum() + abs_beta.sum() * (num_objects - 1)

def _triu(a, sparse):
    if sparse:
        return sp.triu(a, k=1)
    return np.triu(a, k=1)
