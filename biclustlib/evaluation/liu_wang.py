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

from .check import check_biclusterings

def liu_wang_match_score(predicted_biclustering, reference_biclustering):
    """Liu & Wang match score.

    Reference
    ---------
    Liu, X., & Wang, L. (2006). Computing the maximum similarity bi-clusters of gene expression data.
    Bioinformatics, 23(1), 50-56.

    Horta, D., & Campello, R. J. G. B. (2014). Similarity measures for comparing biclusterings.
    IEEE/ACM Transactions on Computational Biology and Bioinformatics, 11(5), 942-954.

    Parameters
    ----------
    predicted_biclustering : biclustlib.model.Biclustering
        Predicted biclustering solution.

    reference_biclustering : biclustlib.model.Biclustering
        Reference biclustering solution.

    Returns
    -------
    lw_match_score : float
        Liu and Wang match score between 0.0 and 1.0.
    """
    check = check_biclusterings(predicted_biclustering, reference_biclustering)

    if isinstance(check, float):
        return check

    k = len(predicted_biclustering.biclusters)

    return sum(max((len(np.intersect1d(bp.rows, br.rows)) + len(np.intersect1d(bp.cols, br.cols))) /
        (len(np.union1d(bp.rows, br.rows)) + len(np.union1d(bp.cols, br.cols)))
        for br in reference_biclustering.biclusters)
        for bp in predicted_biclustering.biclusters) / k
