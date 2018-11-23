from ._base import RBiclustWrapper
import numpy as np

class RConservedGeneExpressionMotifs(RBiclustWrapper):
    """
    """

    def __init__(self, num_biclusters=10, num_seeds=10, num_sets=1000, set_size=7, alpha=0.05):
        super().__init__(data_type=np.int)
        self.num_biclusters = num_biclusters
        self.num_seeds = num_seeds
        self.num_sets = num_sets
        self.set_size = set_size
        self.alpha = alpha

    def _get_parameters(self):
        return {'method' : 'BCXmotifs',
                'ns' : self.num_seeds,
                'nd' : self.num_sets,
                'sd' : self.set_size,
                'alpha' : self.alpha,
                'number' : self.num_biclusters}

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.num_seeds <= 0:
            raise ValueError("num_seeds must be > 0, got {}".format(self.num_seeds))

        if self.num_sets <= 0:
            raise ValueError("num_sets must be > 0, got {}".format(self.num_sets))

        if self.set_size <= 0:
            raise ValueError("set_size must be > 0, got {}".format(self.set_size))

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be >= 0.0 and <= 1.0, got".format(self.alpha))
