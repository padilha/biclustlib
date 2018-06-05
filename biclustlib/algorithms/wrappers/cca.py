from ._base import RBiclustWrapper

class RChengChurchAlgorithm(RBiclustWrapper):
    """Cheng and Church's Algorithm (CCA)

    CCA searches for maximal submatrices with a Mean Squared Residue value below a pre-defined threshold.

    This class is a simple wrapper for the biclust R package
    (https://cran.r-project.org/web/packages/biclust/index.html).

    Reference
    ----------
    Cheng, Y., & Church, G. M. (2000). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found ('number' parameter in the biclust package).

    msr_threshold : float or str, default: 'estimate'
        Maximum mean squared residue accepted ('delta' parameter in the biclust package).

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns ('alpha' parameter in the biclust package).
    """

    def __init__(self, num_biclusters=10, msr_threshold=1.0, multiple_node_deletion_threshold=1.2):
        super().__init__()
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.multiple_node_deletion_threshold = multiple_node_deletion_threshold

    def _get_parameters(self):
        return {'method' : 'BCCC',
                'delta' : self.msr_threshold,
                'alpha' : self.multiple_node_deletion_threshold,
                'number' : self.num_biclusters}

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be >= 0.0, got {}".format(self.msr_threshold))

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError("multiple_node_deletion_threshold must be >= 1.0, got {}".format(self.multiple_node_deletion_threshold))
