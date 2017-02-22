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
from ..models import Bicluster, Biclustering
from itertools import combinations

import numpy as np

class BiCorrelationClusteringAlgorithm(BaseBiclusteringAlgorithm):
    """Bi-Correlation Clustering Algorithm (BCCA)

    BCCA searches for biclusters containing subsets of objects with similar behaviors
    over subsets of features. This algorithm uses the Pearson correlation coefficient
    for measuring the similarity between two objects.

    Reference
    ----------
    Bhattacharya, A. and De, R. K. (2009). Bi-correlation clustering algorithm for determining a
    set of co-regulated genes. Bioinformatics, 25(21):2795-2801.

    Parameters
    ----------
    corr_thr : float, default: 0.9
        Correlation threshold for the final biclusters.

    min_cols : int, default: 3
        Minimum number of columns allowed in the final biclusters.
    """

    def __init__(self, corr_thr=0.9, min_cols=3):
        self.corr_thr = corr_thr
        self.min_cols = min_cols

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        self._validate_parameters()

        num_rows, num_cols = data.shape
        biclusters = []

        for i, j in combinations(range(num_rows), 2):
            cols, corr = self._find_cols(data[i], data[j])

            if len(cols) >= self.min_cols and corr >= self.corr_thr:
                rows = [i, j]

                for k, r in enumerate(data):
                    if k != i and k != j and self._accept(data, rows, cols, r):
                        rows.append(k)

                b = Bicluster(rows, cols)

                if not self._exists(biclusters, b):
                    biclusters.append(b)

        return Biclustering(biclusters)

    def _find_cols(self, ri, rj):
        """Finds the column subset for which the correlation between ri and rj
        stands above the correlation threshold.
        """
        cols = np.arange(len(ri), dtype=np.int)
        corr = self._corr(ri, rj)

        while corr < self.corr_thr and len(cols) >= self.min_cols:
            imax = self._find_max_decrease(ri, rj, cols)
            cols = np.delete(cols, imax)
            corr = self._corr(ri[cols], rj[cols])

        return cols, corr

    def _find_max_decrease(self, ri, rj, indices):
        """Finds the column which deletion causes the maximum increase in
        the correlation value between ri and rj
        """
        kmax, greater = -1, float('-inf')

        for k in range(len(indices)):
            ind = np.concatenate((indices[:k], indices[k+1:]))
            result = self._corr(ri[ind], rj[ind])

            if result > greater:
                kmax, greater = k, result

        return kmax

    def _accept(self, data, rows, cols, r):
        """Checks if row r satisfies the correlation threshold."""
        for i in rows:
            corr = self._corr(r, data[i, cols])

            if corr < self.corr_thr:
                return False

        return True

    def _corr(self, v, w):
        """Calculates the Pearson correlation and returns its absolute value."""
        vc = v - np.mean(v)
        wc = w - np.mean(w)

        x = np.sum(vc * wc)
        y = np.sum(vc * vc) * np.sum(wc * wc)

        return np.abs(x / np.sqrt(y))

    def _exists(self, biclusters, bic):
        """Checks if a bicluster has already been found."""
        for b in biclusters:
            if len(b.rows) == len(bic.rows) and len(b.cols) == len(bic.cols) and \
                 np.all(b.rows == bic.rows) and np.all(b.cols == bic.cols):
                return True
        return False

    def _validate_parameters(self):
        if self.corr_thr < 0.0 or self.corr_thr > 1.0:
            raise ValueError("'corr_thr' value must be between 0.0 and 1.0")
        if self.min_cols < 3:
            raise ValueError("'min_cols' value must be a positive integer greater than or equal to 3")
