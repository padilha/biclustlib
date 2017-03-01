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

import numpy as np

class ConservedGeneExpressionMotifs(BaseBiclusteringAlgorithm):
    """Conserved Gene Expression Motifs (xMOTIFs)

    xMOTIFs is a nondeterministic algorithm that finds submatrices with simultaneously
    conserved rows in subsets of columns in a discrete data matrix.

    Reference
    ----------
    Murali, T. and Kasif, S. (2003). Extracting conserved gene expression motifs from gene expression
    data. In Pacific Symposium on Biocomputing, volume 8, pages 77-88.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found.

    num_seeds : int, default: 10
        Number of seed columns chosen for each bicluster.

    num_sets : int, default: 1000
        Number of discriminating sets generated for each seed column.

    set_size : int, default: 7
        Size of each discriminating set.

    alpha : float, default: 0.05
        Minimum fraction of dataset columns that a bicluster must satisfy.
    """

    def __init__(self, num_biclusters=10, num_seeds=10, num_sets=1000, set_size=7, alpha=0.05):
        self.num_biclusters = num_biclusters
        self.num_seeds = num_seeds
        self.num_sets = num_sets
        self.set_size = set_size
        self.alpha = alpha

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        self._validate_parameters()
        self._validate_data(data)

        num_remaining_rows, num_cols = data.shape
        remaining_rows = np.ones(num_remaining_rows, np.bool)
        biclusters = []

        for i in range(self.num_biclusters):
            indices = np.where(remaining_rows)[0]
            b = self._find_motif(data, indices)
            biclusters.append(b)

            remaining_rows[b.rows] = False
            num_remaining_rows -= len(b.rows)

            if num_remaining_rows == 0:
                break

        return Biclustering(biclusters)

    def _find_motif(self, data, row_indices):
        """Finds the largest xMOTIF (this is the direct implementation of the
        pseucode of the FindMotif() procedure described in the original paper).
        """
        num_rows, num_cols = data.shape
        best_motif = Bicluster([], [])
        seeds = np.random.choice(num_cols, self.num_seeds, replace=False)

        for s in seeds:
            seed_col = data[row_indices, s][:, np.newaxis]

            for i in range(self.num_sets):
                cols_set = np.random.choice(num_cols, self.set_size, replace=False)

                rows_comp_data = seed_col == data[np.ix_(row_indices, cols_set)]
                selected_rows = np.array([y for x, y in enumerate(row_indices) if np.all(rows_comp_data[x])], np.int)

                seed_values = data[selected_rows, s][:, np.newaxis]
                cols_comp_data = seed_values == data[selected_rows]
                selected_cols = np.array([k for k in range(num_cols) if np.all(cols_comp_data[:, k])])

                if len(selected_cols) >= self.alpha * num_cols and len(selected_rows) > len(best_motif.rows):
                    best_motif = Bicluster(selected_rows, selected_cols)

        return best_motif

    def _validate_parameters(self):
        if self.num_biclusters <= 0 or self.num_seeds <= 0 or self.num_sets <= 0 or self.set_size <= 0:
            raise ValueError("'num_biclusters', 'num_seeds', 'num_sets' and 'set_size' must be greater than zero")

        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("'alpha' must assume a value between 0.0 and 1.0")

    def _validate_data(self, data):
        if not issubclass(data.dtype.type, np.integer):
            raise ValueError('XMotifs requires discrete data')
