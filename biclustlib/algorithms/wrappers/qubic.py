from ._base import BaseExecutableWrapper
from ...models import Bicluster, Biclustering
from os.path import dirname, join

import numpy as np
import re

class QualitativeBiclustering(BaseExecutableWrapper):

    def __init__(self, num_biclusters=10, ranks=1, quant=0.06, consistency=0.95, max_overlap_level=1.0, tmp_dir='.qubic_tmp'):
        module_dir = dirname(__file__)

        exec_comm = join(module_dir, 'bin', 'qubic') + \
                    ' -i {_data_filename}' + \
                    ' -q {quant}' + \
                    ' -r {ranks}' + \
                    ' -f {max_overlap_level}' + \
                    ' -c {consistency}' + \
                    ' -o {num_biclusters}'

        super().__init__(exec_comm, tmp_dir=tmp_dir)

        self.num_biclusters = num_biclusters
        self.ranks = ranks
        self.quant = quant
        self.consistency = consistency
        self.max_overlap_level = max_overlap_level

        self._data_filename = 'data.txt'
        self._output_filename = 'data.txt.blocks'

    def _parse_output(self):
        with open(self._output_filename, 'r') as f:
            content = f.read()
            bc_strings = re.split('BC[0-9]+', content)[1:]

        return Biclustering([self._parse_bicluster(b) for b in bc_strings])

    def _parse_bicluster(self, string):
        after_genes = re.split('Genes \[[0-9]+\]:', string).pop()
        genes, after_conds = re.split('Conds \[[0-9]+\]:', after_genes)
        genes = genes.split()
        conds = after_conds.split('\n')[0].split()
        return Bicluster(self._convert(genes), self._convert(conds))

    def _convert(self, str_array):
        return np.array([int(a.split('_').pop()) for a in str_array])

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.ranks <= 0:
            raise ValueError("ranks must be > 0, got {}".format(self.ranks))

        if self.quant <= 0.0 or self.quant >= 1.0:
            raise ValueError("quant must be > 0.0 and < 1.0, got {}".format(self.quant))

        if self.consistency <= 0.0 or self.consistency >= 1.0:
            raise ValueError("consistency must be > 0.0 and < 1.0, got {}".format(self.consistency))

        if self.max_overlap_level <= 0.0 or self.max_overlap_level >= 1.0:
            raise ValueError("max_overlap_level must be > 0.0 and < 1.0, got {}".format(self.max_overlap_level))
