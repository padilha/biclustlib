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

from ._base import ExecutableWrapper
from ._util import parse_in_chunks
from os.path import dirname, join
from ...models import Bicluster, Biclustering

import os
import numpy as np

class OrderPreservingSubMatrix(ExecutableWrapper):
    """Order-Preserving SubMatrix (OPSM)

    OPSM finds biclusters, each one containing rows that follow the same order under the
    bicluster columns.

    Reference
    ---------
    Ben-Dor, A., Chor, B., Karp, R., and Yakhini, Z. (2003). Discovering local structure in gene expression
    data: the order-preserving submatrix problem. Journal of computational biology, 10(3-4), 373-384.

    Parameters
    ----------
    num_best_partial_models : int, default: 100
        Number of best partial models to maintain from one iteration to another.
    """

    def __init__(self, num_best_partial_models=100):
        super().__init__()
        self.num_best_partial_models = num_best_partial_models

    def _get_command(self, data, data_path, output_path):
        module_dir = dirname(__file__)
        num_rows, num_cols = data.shape

        return 'java -jar ' + \
               join(module_dir, 'jar', 'opsm', 'OPSM.jar') + \
               ' {}'.format(data_path) + \
               ' {}'.format(num_rows) + \
               ' {}'.format(num_cols) + \
               ' {}'.format(output_path) + \
               ' {}'.format(self.num_best_partial_models)

    def _write_data(self, data_path, data):
        with open(data_path, 'wb') as f:
            np.savetxt(f, data, delimiter='\t')

    def _parse_output(self, output_path):
        biclusters = []

        if os.path.exists(output_path):
            for rows, cols in parse_in_chunks(output_path, chunksize=3, rows_idx=0, cols_idx=1):
                b = Bicluster(rows, cols)
                biclusters.append(b)

        return Biclustering(biclusters)

    def _validate_parameters(self):
        if self.num_best_partial_models <= 0:
            raise ValueError('num_best_partial_models must be > 0, got {}'.format(self.num_best_partial_models))
