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
from ...models import Bicluster, Biclustering
from os.path import dirname, join

import numpy as np
import os, re

class BayesianBiclustering(ExecutableWrapper):
    """Bayesian BiClustering (BBC)

    BBC assumes the plaid model and uses a Gibbs sampling procedure for its statistical inference.

    This class is a simple wrapper for the executable obtained after compiling the C code
    provided by the original authors of this algorithm (available in http://www.people.fas.harvard.edu/~junliu/BBC/).
    The binaries contained in this package were compiled for the x86_64 architecture.

    Reference
    ---------
    Gu, J., and Liu, J. S. (2008). Bayesian biclustering of gene expression data. BMC genomics, 9(1), S4.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found.

    normalization : str, default: 'iqrn'
        Normalization method used by the algorithm. Must be one of ('iqrn', 'sqrn', 'csn', 'rsn')
        (see http://www.people.fas.harvard.edu/~junliu/BBC/BBC_manual.pdf for details).

    alpha : float, default: 90.0
        Alpha value for the normalization step (used only when normalization is
        'iqrn' or 'sqrn').
    """

    def __init__(self, num_biclusters=10, normalization='iqrn', alpha=90):
        super().__init__()
        self.num_biclusters = num_biclusters
        self.normalization = normalization
        self.alpha = alpha

    def _get_command(self, data_path, output_path):
        module_dir = dirname(__file__)

        return join(module_dir, 'bin', 'bbc', 'BBC') + \
               ' -i {data_path}' + \
               ' -k {num_biclusters}' + \
               ' -o {output_path}' + \
               ' -n {normalization}' + \
               ' -r {alpha}'.format(data_path=data_path,
                                    output_path=output_path,
                                    **self.__dict__)

    def _parse_output(self, output_path):
        biclusters = []

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                content = f.read()
                biclusters_str = re.split('bicluster[0-9]+\n', content)[1:]

                for b_str in biclusters_str:
                    row_matches = re.findall('ROW_[0-9]+', b_str)
                    rows = np.array([r.split('_')[1] for r in row_matches], dtype=np.int)

                    col_matches = re.findall('COL_[0-9]+', b_str)
                    cols = np.array([c.split('_')[1] for c in col_matches], dtype=np.int)

                    biclusters.append(Bicluster(rows, cols))

        return Biclustering(biclusters)

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        norm = ('iqrn', 'sqrn', 'csn', 'rsn')

        if self.normalization not in norm:
            raise ValueError("normalization must be one of {}, got {}".format(norm, self.normalization))

        if self.alpha <= 0.0 or self.alpha >= 100.0:
            raise ValueError("alpha must be > 0 and < 100, got {}".format(self.alpha))
