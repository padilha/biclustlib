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

from time import sleep
from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import check_array

from .._base import BaseBiclusteringAlgorithm
from ...models import Bicluster, Biclustering

import os
import shutil
import numpy as np

class ExecutableWrapper(BaseBiclusteringAlgorithm, metaclass=ABCMeta):
    """This class defines the skeleton of a naive executable wrapper. In summary,
    in every execution, it will create a temporary directory, save the input data
    as a txt file, run the wrapped algorithm and parse the output files. Finally,
    the temporary directory will be removed.
    """

    def __init__(self, exec_comm, tmp_dir, sleep=True):
        super().__init__()

        self.__exec_comm = exec_comm
        self.__sleep = sleep

        self._data_filename = 'data.txt'
        self._output_filename = 'output.txt'

        if not tmp_dir.startswith('.'):
            self.__tmp_dir = '.' + tmp_dir
        else:
            self.__tmp_dir = tmp_dir

    def run(self, data):
        data = check_array(data, dtype=np.double, copy=True)

        self._validate_parameters()

        if self.__sleep:
            sleep(1)

        self.__change_working_dir()
        self._write_data(data)
        os.system(self.__exec_comm.format(**self.__dict__))
        biclustering = self._parse_output()
        self.__restore_working_dir()

        return biclustering

    def __change_working_dir(self):
        os.mkdir(self.__tmp_dir)
        os.chdir(self.__tmp_dir)

    def __restore_working_dir(self):
        os.chdir('..')
        shutil.rmtree(self.__tmp_dir)

    def _write_data(self, data, header=True, row_names=True):
        num_rows, num_cols = data.shape

        if row_names:
            row_names = np.char.array(['GENE_' + str(i) for i in range(num_rows)])[:, np.newaxis]
            data = np.hstack((row_names, data))

        if header:
            header = 'GENES\t' + '\t'.join('COND_' + str(i) for i in range(num_cols))
        else:
            header = ''

        with open(self._data_filename, 'wb') as f:
            np.savetxt(f, data, delimiter='\t', header=header, fmt='%s', comments='')

    @abstractmethod
    def _parse_output(self):
        pass


class BicatWrapper(ExecutableWrapper, metaclass=ABCMeta):
    """This class defines the skeleton of a wrapper for the Biclustering
    Analysis Toolbox BicAT, available in http://people.ee.ethz.ch/~sop/bicat/.
    """

    def __init__(self, exec_comm, tmp_dir, sleep=True):
        super().__init__('java -jar ' + exec_comm, tmp_dir, sleep)

    def _write_data(self, data):
        super()._write_data(data, header=False, row_names=False)

    def _parse_output(self):
        biclusters = []

        if os.path.exists(self._output_filename):
            with open(self._output_filename, 'r') as f:
                all_lines = f.readlines()
                rows, cols = None, None

                for i, line in enumerate(all_lines):
                    if i % 2 == 0:
                        rows = self._convert(line.split())
                    else:
                        cols = self._convert(line.split())
                        biclusters.append(Bicluster(rows, cols))

        return Biclustering(biclusters)

    def _convert(self, bit_array):
        return np.array([int(i) for i, n in enumerate(bit_array) if int(n)])


class SklearnWrapper(BaseBiclusteringAlgorithm, metaclass=ABCMeta):
    """This class defines the skeleton of a wrapper for the scikit-learn
    package.
    """

    def __init__(self, constructor, **kwargs):
        self.wrapped_algorithm = constructor(**kwargs)

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        self.algorithm.fit(data)

        biclusters = []

        for rows, cols in zip(*self.wrapped_algorithm.biclusters_):
            if rows.dtype == np.bool and cols.dtype == np.bool:
                rows = np.nonzero(rows)
                cols = np.nonzero(cols)

            if len(rows) and len(cols):
                b = Bicluster(rows, cols)
                biclusters.append(b)

        return Biclustering(biclusters)
