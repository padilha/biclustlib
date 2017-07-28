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

from abc import ABCMeta, abstractmethod

class BaseBiclusteringAlgorithm(object, metaclass=ABCMeta):
    """A class that defines the skeleton of a biclustering algorithm implementation."""

    @abstractmethod
    def run(self, data):
        """Method needed to run a biclustering algorithm."""
        pass

    @abstractmethod
    def _validate_parameters(self):
        """Method to validate the input parameters of a biclustering algorithm, if necessary."""
        pass
