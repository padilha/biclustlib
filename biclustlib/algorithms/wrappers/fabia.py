from ._base import SklearnWrapper
from fabia import FabiaBiclustering
from ...models import Bicluster

import numpy as np

class FactorAnalysisForBiclusterAcquisition(SklearnWrapper):
    """Fabia Biclustering.

    Fabia is an algorithm that assumes a multiplicative model and uses a factor analysis approach
    to fit such a model on the input data.

    This class is a simple wrapper of Fabia's implementation available in
    the fabia package (https://github.com/untom/pyfabia).

    Reference
    ---------
    Hochreiter, S., Bodenhofer, U., Heusel, M., Mayr, A., Mitterecker, A., Kasim, A., Khamiakova, T.,
    Sanden, S. V., Lin, D., Talloen, W., Bijnens, L., Gohlmann, H. W. H., Shkedy, Z., and Clevert, D. A. (2010).
    FABIA: factor analysis for bicluster acquisition. Bioinformatics, 26(12), 1520-1527.

    Parameters
    ----------
    **kwargs: dict
        See fabia.FabiaBiclustering documentation. Note that in this class the parameters must be passed
        as keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(FabiaBiclustering, **kwargs)
        self.__data_estimation = None

    def run(self, data):
        self.__data_estimation = None
        return super().run(data)

    def _get_bicluster(self, rows, cols):
        if self.__data_estimation is None:
            self.__data_estimation = np.dot(self.wrapped_algorithm.Z_, self.wrapped_algorithm.L_)
        return Bicluster(rows, cols, self.__data_estimation[np.ix_(rows, cols)])

    def _validate_parameters(self):
        """This Fabia wrapper does not require any data parameters validation step.
        Refer to the _validate_params method of fabia.FabiaBiclustering."""
        pass
