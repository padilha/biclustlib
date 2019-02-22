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

import numpy as np

class Bicluster:
    """This class models a bicluster.

    Parameters
    ----------
    rows : numpy.array
        Rows of the bicluster (assumes that row indexing starts at 0).

    cols : numpy.array
        Columns of the bicluster (assumes that column indexing starts at 0).
    """

    def __init__(self, rows, cols):
        if rows.dtype == np.bool and cols.dtype == np.bool:
            self.rows = np.nonzero(rows)[0]
            self.cols = np.nonzero(cols)[0]
        elif rows.dtype == np.int and cols.dtype == np.int:
            self.rows = rows
            self.cols = cols
        else:
            raise ValueError("rows and cols must be bool or int numpy.arrays")

    def intersection(self, other):
        """Returns a bicluster that represents the area of overlap between two biclusters."""
        rows_intersec = np.intersect1d(self.rows, other.rows)
        cols_intersec = np.intersect1d(self.cols, other.cols)
        return Bicluster(rows_intersec, cols_intersec)

    def union(self, other):
        rows_union = np.union1d(self.rows, other.rows)
        cols_union = np.union1d(self.cols, other.cols)
        return Bicluster(rows_union, cols_union)

    def overlap(self, other):
        min_area = min(self.area, other.area)
        return self.intersection(other).area / min_area

    @property
    def area(self):
        """Calculates the number of matrix elements of the bicluster."""
        return len(self.rows) * len(self.cols)

    def sort(self):
        """Sorts the array of row and the array of column indices of the bicluster."""
        self.rows.sort()
        self.cols.sort()

    def __str__(self):
        return 'Bicluster(rows={0}, cols={1})'.format(self.rows, self.cols)


class Biclustering:
    """This class models a biclustering.

    Parameters
    ----------
    biclusters : list
        A list of instances from the Bicluster class.
    """

    def __init__(self, biclusters):
        if all(isinstance(b, Bicluster) for b in biclusters):
            self.biclusters = biclusters
        else:
            raise ValueError("biclusters list contains an element that is not a Bicluster instance")

    def __str__(self):
        return '\n'.join(str(b) for b in self.biclusters)
