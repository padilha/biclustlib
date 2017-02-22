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

from _base import BaseBiclusteringAlgorithm
from ..models import Bicluster, Biclustering

import numpy as np

class ChengChurchAlgorithm(BaseBiclusteringAlgorithm):
    """Cheng and Church's Algorithm (CCA)

    CCA searches for maximal submatrices with a Mean Squared Residue value below a pre-defined threshold.

    Reference
    ----------
    Cheng, Y. and Church, G. M. (2000). Biclustering of expression data. In Proceedings of the 8th
    International Conference on Intelligence Systems for Molecular Biology, volume 8, pages 93-103.
    AAAI Press.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found.

    msr_threshold : float, default: 0.1
        Maximum mean squared residue accepted.

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns.

    data_min_cols : int, default: 100
        Minimum number of dataset columns required to perform multiple column deletion.
    """
    def __init__(self, num_biclusters=10, msr_threshold=0.1, multiple_node_deletion_threshold=1.2, data_min_cols=100):
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.multiple_node_deletion_threshold = multiple_node_deletion_threshold
        self.data_min_cols = data_min_cols

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        self._validate_parameters()

        data = np.copy(data)

        num_rows, num_cols = data.shape
        min_value = np.min(data)
        max_value = np.max(data)

        biclusters = []

        for i in range(self.num_biclusters):
            rows = np.ones(num_rows, dtype=np.bool)
            cols = np.ones(num_cols, dtype=np.bool)

            self._multiple_node_deletion(data, rows, cols)
            self._single_node_deletion(data, rows, cols)
            self._node_addition(data, rows, cols)

            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            # masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (len(row_indices), len(col_indices))
                data[rows[:, np.newaxis], cols] = np.random.uniform(low=min_value, high=max_value, shape=bicluster_shape)

            biclusters.append(Bicluster(row_indices, col_indices))

        return Biclustering(biclusters)

    def _single_node_deletion(self, data, rows, cols):
        """Performs the single row/column deletion step (this is a direct implementation
        of Algorithm 1 described in the original paper)"""
        squared_residues = self._calculate_squared_residues(data[rows][:, cols])
        msr = np.mean(squared_residues)

        while msr > self.msr_threshold:
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            row_msr = np.mean(squared_residues, axis=1)
            col_msr = np.mean(squared_residues, axis=0)

            row2remove = np.argmax(row_msr)
            col2remove = np.argmax(col_msr)

            if row_msr[row2remove] >= col_msr[col2remove]:
                rows[row_indices[row2remove]] = False
            else:
                cols[col_indices[col2remove]] = False

            squared_residues = self._calculate_squared_residues(data[rows][:, cols])
            msr = np.mean(squared_residues)

    def _multiple_node_deletion(self, data, rows, cols):
        """Performs the multiple row/column deletion step (this is a direct implementation
        of Algorithm 2 described in the original paper)"""
        squared_residues = self._calculate_squared_residues(data)
        msr = np.mean(squared_residues)

        while msr > self.msr_threshold:
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            row_msr = np.mean(squared_residues, axis=1)
            rows2remove = np.where(row_msr > self.multiple_node_deletion_threshold * msr)[0]
            rows[row_indices[rows2remove]] = False

            if len(cols) >= self.data_min_cols:
                col_msr = np.mean(squared_residues, axis=0)
                cols2remove = np.where(col_msr > self.multiple_node_deletion_threshold * msr)[0]
                cols[col_indices[cols2remove]] = False
            else:
                cols2remove = np.array([])

            if len(rows2remove) == 0 and len(cols2remove) == 0:
                break

            squared_residues = self._calculate_squared_residues(data[rows][:, cols])
            msr = np.mean(squared_residues)

    def _node_addition(self, data, rows, cols):
        """Performs the row/column addition step (this is a direct implementation
        of Algorithm 3 described in the original paper)"""
        added_cols, added_rows, added_inverted_rows = True, True, True

        while added_cols or added_rows or added_inverted_rows:
            added_cols = self._add_cols(data, rows, cols)
            added_rows = self._add_rows(data, rows, cols)
            added_inverted_rows = self._add_rows(-data, rows, cols)

    def _add_rows(self, data, rows, cols):
        """Selects the rows to be included in the bicluster."""
        removed_rows = np.logical_not(rows)
        removed_row_indices = np.where(removed_rows)[0]
        removed_row_squared_residues = self._calculate_squared_residues(data[removed_rows][:, cols])
        removed_row_msr = np.mean(removed_row_squared_residues, axis=1)
        rows2add = np.where(removed_row_msr <= self.msr_threshold)[0]
        rows[removed_row_indices[rows2add]] = True
        return len(rows2add) > 0

    def _add_cols(self, data, rows, cols):
        """Selects the columns to be included in the bicluster."""
        removed_cols = np.logical_not(cols)
        removed_col_indices = np.where(removed_cols)[0]
        removed_col_squared_residues = self._calculate_squared_residues(data[rows][:, removed_cols])
        removed_col_msr = np.mean(removed_col_squared_residues, axis=0)
        cols2add = np.where(removed_col_msr <= self.msr_threshold)[0]
        cols[removed_col_indices[cols2add]] = True
        return len(cols2add) > 0

    def _calculate_squared_residues(self, data):
        """Calculate the elements' squared residues of a data matrix."""
        data_mean = np.mean(data)
        row_means = np.mean(data, axis=1)
        col_means = np.mean(data, axis=0)
        residues = data - row_means[:, np.newaxis] - col_means + data_mean
        return residues * residues

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("'num_biclusters' must be greater than zero")

        if self.msr_threshold < 0.0:
            raise ValueError("'msr_threshold' must be greater than or equal to zero")

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError("'multiple_node_deletion_threshold' must be greater than or equal to 1")
