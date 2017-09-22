from ._base import BaseExecutableWrapper
from ...models import Bicluster, Biclustering
from os.path import dirname, join

import numpy as np
import re

class BayesianBiclustering(BaseExecutableWrapper):

    def __init__(self, num_biclusters=10, normalization='iqrn', alpha=90, tmp_dir='.bbc_tmp'):
        module_dir = dirname(__file__)

        exec_comm = join(module_dir, 'bin', 'BBC') + \
                    ' -i {_data_filename}' + \
                    ' -k {num_biclusters}' + \
                    ' -o {_output_filename}' + \
                    ' -n {normalization}' + \
                    ' -r {alpha}'

        super().__init__(exec_comm, tmp_dir=tmp_dir)

        self.num_biclusters = num_biclusters
        self.normalization = normalization
        self.alpha = alpha

        self._data_filename = 'data.txt'
        self._output_filename = 'output.txt'

    def _parse_output(self):
        with open(self._output_filename, 'r') as f:
            content = f.read()
            biclusters_str = re.split('bicluster[0-9]+\n', content)

            bic_str = biclusters_str.pop(0)
            bic = float(bic_str.rstrip().split('\t')[-1].split()[1])

            biclusters = []

            for b_str in biclusters_str:
                rows_str, cols_str = b_str.split('col\t')

                rows_str = rows_str.split('\n')[2:-1]
                rows = np.array([int(x.split('\t')[0]) - 1 for x in rows_str])

                cols_str = cols_str.split('\n')[1:-1]
                cols = np.array([int(x.split('\t')[0]) - 1 for x in cols_str])

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
