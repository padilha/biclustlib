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
        Normalization method used by the algorithm. Must be one of ('iqrn', 'sqrn', 'csn', 'rsn', 'none')
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

    def _get_command(self, data, data_path, output_path):
        module_dir = dirname(__file__)

        return join(module_dir, 'bin', 'bbc', 'BBC') + \
               ' -i {}'.format(data_path) + \
               ' -k {}'.format(self.num_biclusters) + \
               ' -o {}'.format(output_path) + \
               ' -n {}'.format(self.normalization) + \
               ' -r {}'.format(self.alpha)

    def _write_data(self, data_path, data):
        header = 'DATA\t' + '\t'.join('COL_' + str(i) for i in range(data.shape[1]))
        row_names = np.char.array(['ROW_' + str(i) for i in range(data.shape[0])])
        data = data.astype(np.str)
        data = np.hstack((row_names[:, np.newaxis], data))

        with open(data_path, 'wb') as f:
            np.savetxt(f, data, delimiter='\t', header=header, fmt='%s', comments='')

    def _parse_output(self, output_path):
        biclusters = []

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                content = f.read()
                biclusters_str = re.split('bicluster[0-9]+\n', content)[1:]

                for b_str in biclusters_str:
                    ground_effect = float(b_str.split('\n', 1)[0].split()[-1])
                    rows, rows_effects = self._get_indices_and_effects(b_str, 'ROW_[0-9]+')
                    cols, cols_effects = self._get_indices_and_effects(b_str, 'COL_[0-9]+')
                    b_data = ground_effect + rows_effects[:, np.newaxis] + cols_effects
                    biclusters.append(Bicluster(rows, cols, b_data))

        return Biclustering(biclusters)

    def _get_indices_and_effects(self, bicluster_string, pattern):
        matches = re.findall(pattern + '\t+\d+\.?\d+', bicluster_string)

        indices = []
        effects = []

        for m in matches:
            i, e = m.split('_')[1].split()
            indices.append(i)
            effects.append(e)

        return np.array(indices, np.int), np.array(effects, np.double)

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        norm = ('iqrn', 'sqrn', 'csn', 'rsn', 'none')

        if self.normalization not in norm:
            raise ValueError("normalization must be one of {}, got {}".format(norm, self.normalization))

        if self.alpha <= 0.0 or self.alpha >= 100.0:
            raise ValueError("alpha must be > 0 and < 100, got {}".format(self.alpha))
