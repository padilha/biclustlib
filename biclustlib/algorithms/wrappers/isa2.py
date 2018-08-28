from ._base import RBiclustWrapper

import numpy as np
import rpy2.robjects as robjs

class IterativeSignatureAlgorithm2(RBiclustWrapper):

    def __init__(self, num_seeds=100, row_thr=2.0, col_thr=2.0):
        super().__init__()
        self.num_seeds = num_seeds
        self.row_thr = row_thr
        self.col_thr = col_thr
        self._r_lib = 'isa2'
        self._r_func = 'isa'

    def _get_parameters(self):
        return {'no.seeds' : self.num_seeds,
                'thr.row' : self.row_thr,
                'thr.col' : self.col_thr}

    def _get_biclustering(self, data, biclust_result):
        isa_biclust = robjs.r['isa.biclust']
        biclust_result = isa_biclust(biclust_result)
        return super()._get_biclustering(data, biclust_result)

    def _validate_parameters(self):
        pass
