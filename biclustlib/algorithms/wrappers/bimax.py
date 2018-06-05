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

from ._base import RBiclustWrapper

import numpy as np

class RBinaryInclusionMaximalBiclusteringAlgorithm(RBiclustWrapper):
    """Binary Inclusion-Maximal Biclustering Algorithm (Bimax)

    Bimax searches for submatrices with all values equal to 1 in a binary matrix.

    Reference
    ---------
    Prelic, A., Bleuler, S., Zimmermann, P., Wille, A., Buhlmann, P., Gruissem, W., Hennig, L., Thiele, L. &
    Zitzler, E. (2006). A systematic comparison and evaluation of biclustering methods for gene expression data.
    Bioinformatics, 22(9), 1122-1129.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found.

    min_rows : int, default: 2
        Minimum number of rows of a bicluster.

    min_cols : int, default: 2
        Minimum number of columns of a bicluster.
    """

    def __init__(self, num_biclusters=10, min_rows=2, min_cols=2):
        super().__init__(data_type=np.bool)
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.num_biclusters = num_biclusters

    def _get_parameters(self):
        return {'method' : 'BCBimax',
                'minr' : self.min_rows,
                'minc' : self.min_cols,
                'number' : self.num_biclusters}

    def _validate_parameters(self):
        if self.min_rows <= 0:
            raise ValueError("num_rows must be > 0, got {}".format(self.num_rows))

        if self.min_cols <= 0:
            raise ValueError("num_cols must be > 0, got {}".format(self.num_cols))

        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))
