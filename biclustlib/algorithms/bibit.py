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
from gmpy import popcount
from sklearn.utils.validation import check_array

import numpy as np

class BitPatternBiclusteringAlgorithm(BaseBiclusteringAlgorithm):
    """Bit Pattern Biclustering Algorithm (BiBit)

    BiBit is an algorithm to extract submatrices from binary datasets.

    Reference
    ----------
    Rodriguez-Baena, D. S., Perez-Pulido, A. J., Aguilar, J. S. (2011). A biclustering algorithm
    for extracting bit-patterns from binary datasets. Bioinformatics, 27(19):2738-2745.

    Parameters
    ----------
    min_rows : int, default: 2
        Minimum number of rows allowed in the final biclusters.

    min_cols : int, default: 2
        Minimum number of columns allowed in the final biclusters.
    """

    def __init__(self, min_rows=2, min_cols=2):
        self.min_rows = min_rows
        self.min_cols = min_cols

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        data = check_array(data, dtype=np.bool, copy=True)
        self._validate_parameters()

        data = [np.packbits(row) for row in data]
        biclusters = []
        patterns_found = set()

        for ri, rj in combinations(data, 2):
            pattern = np.bitwise_and(ri, rj)
            pattern_cols = sum(popcount(int(n)) for n in pattern)

            if pattern_cols >= self.min_cols and self._is_new(patterns_found, pattern):
                rows = [k for k, r in enumerate(data) if self._match(pattern, r)]

                if len(rows) >= self.min_rows:
                    cols = np.where(np.unpackbits(pattern) == 1)[0]
                    biclusters.append(Bicluster(rows, cols))

        return Biclustering(biclusters)

    def _match(self, pattern, row):
        """Checks if a row matches a pattern."""
        result = np.bitwise_and(pattern, row)
        return np.all(result == pattern)

    def _is_new(self, patterns_set, pattern):
        """Checks if a pattern has not already been found. If it is new, inserts
        it in the patterns set."""
        p = tuple(pattern)
        if p not in patterns_set:
            patterns_set.add(p)
            return True
        return False

    def _validate_parameters(self):
        if self.min_rows < 2:
            raise ValueError("min_rows must be >= 2, got {}".format(self.min_rows))

        if self.min_cols < 2:
            raise ValueError("min_cols must be >= 2, got {}".format(self.min_cols))
