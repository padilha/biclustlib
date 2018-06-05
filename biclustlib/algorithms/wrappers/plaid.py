from ._base import RBiclustWrapper

class RPlaid(RBiclustWrapper):
    """Plaid biclustering algorithm.

    This algorithm fits the plaid model using a binary least squares approach.

    This class is a simple wrapper for the biclust R package
    (https://cran.r-project.org/web/packages/biclust/index.html).

    Reference
    ----------
    Turner, H., Bailey, T., & Krzanowski, W. (2005). Improved biclustering of microarray data demonstrated
    through systematic performance tests. Computational statistics & data analysis, 48(2), 235-254.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found ('max.layers' parameter in the biclust package).

    fit_background_layer : bool, default: True
        If True fits a background layer which represents common effects of all elements of the data matrix
        ('background' parameter in the biclust package).

    row_prunning_threshold : float, default: 0.5
        Threshold for row prunning ('row.release' parameter in the biclust package).

    col_prunning_threshold : float, default: 0.5
        Threshold for column prunning ('col.release' parameter in the biclust package).

    significance_tests : int, default: 0
        Number of significance tests to be performed for a layer ('shuffle' parameter in the biclust package).

    back_fitting_steps : int, default: 1
        Number of back fitting steps ('back.fit' parameter in the biclust package).

    initialization_iterations : int, default: 6
        Number of k-means runs to initialize a new layer ('iter.startup' parameter in the biclust package).

    iterations_per_layer : int, default: 10
        Number of prunning iterations per layer ('iter.layer' parameter in the biclust package).
    """
    def __init__(self, num_biclusters=10, fit_background_layer=True, row_prunning_threshold=0.7,
                 col_prunning_threshold=0.7, significance_tests=0, back_fitting_steps=1,
                 initialization_iterations=6, iterations_per_layer=10):
        super().__init__()
        self.num_biclusters = num_biclusters
        self.fit_background_layer = fit_background_layer
        self.row_prunning_threshold = row_prunning_threshold
        self.col_prunning_threshold = col_prunning_threshold
        self.significance_tests = significance_tests
        self.back_fitting_steps = back_fitting_steps
        self.initialization_iterations = initialization_iterations
        self.iterations_per_layer = iterations_per_layer

    def _get_parameters(self):
        return {'method' : 'BCPlaid',
                'background' : self.fit_background_layer,
                'row.release' : self.row_prunning_threshold,
                'col.release' : self.col_prunning_threshold,
                'shuffle' : self.significance_tests,
                'back.fit' : self.back_fitting_steps,
                'max.layers' : self.num_biclusters,
                'iter.startup' : self.initialization_iterations,
                'iter.layer' : self.iterations_per_layer,
                'verbose' : False}

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if not isinstance(self.fit_background_layer, bool):
            raise ValueError("fit_background_layer must be either True or False, got {}".format(self.fit_background_layer))

        if self.initialization_iterations <= 0:
            raise ValueError("initialization_iterations must be > 0, got {}".format(self.initialization_iterations))

        if self.iterations_per_layer <= 0:
            raise ValueError("iterations_per_layer must be > 0, got {}".format(self.iterations_per_layer))

        if self.significance_tests < 0:
            raise ValueError("significance_tests must be >= 0, got {}".format(self.significance_tests))

        if self.back_fitting_steps < 0:
            raise ValueError("back_fitting_steps must be >= 0, got {}".format(self.back_fitting_steps))

        if not (0.0 < self.row_prunning_threshold < 1.0):
            raise ValueError("row_prunning_threshold must be > 0.0 and < 1.0, got {}".format(self.row_prunning_threshold))

        if not (0.0 < self.col_prunning_threshold < 1.0):
            raise ValueError("col_prunning_threshold must be > 0.0 and < 1.0, got {}".format(self.col_prunning_threshold))
