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

from ._base import SklearnWrapper
from sklearn.cluster.bicluster import SpectralBiclustering

class Spectral(SklearnWrapper):
    """Spectral Biclustering.

    Spectral is an algorithm that uses the singular value decomposition to find biclusterings with
    checkerboard structures.

    This class is a simple wrapper of Spectral's implementation available in
    the scikit-learn package.

    Reference
    ---------
    Kluger, Y., Basri, R., Chang, J. T., and Gerstein, M. (2003). Spectral biclustering of microarray data:
    coclustering genes and conditions. Genome research, 13(4), 703-716.

    Parameters
    ----------
    **kwargs: dict
        See sklearn.cluster.bicluster.SpectralBiclustering documentation. Note that in this class the parameters must be passed
        as keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(SpectralBiclustering, **kwargs)

    def _validate_parameters(self):
        """This Spectral wrapper does not require any data parameters validation step.
        Refer to the _check_parameters method of sklearn.cluster.bicluster.SpectralBiclustering."""
        pass
