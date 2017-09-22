from time import sleep
from abc import ABCMeta, abstractmethod
from .._base import BaseBiclusteringAlgorithm

import os
import shutil
import numpy as np

class BaseExecutableWrapper(BaseBiclusteringAlgorithm, metaclass=ABCMeta):

    def __init__(self, exec_comm, tmp_dir, sleep=True):
        super().__init__()

        self.__exec_comm = exec_comm
        self.__sleep = sleep

        self._data_filename = None
        self._output_filename = None

        if not tmp_dir.startswith('.'):
            self.__tmp_dir = '.' + tmp_dir
        else:
            self.__tmp_dir = tmp_dir

    def run(self, data):
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


class BicatWrapper(BaseExecutableWrapper, metaclass=ABCMeta):

    def __init__(self, exec_comm, tmp_dir, data_filename, output_filename, sleep=True):
        super().__init__(exec_comm, tmp_dir, data_filename, output_filename, sleep=True)

    def _write_data(self, data, filename):
        super()._write_data(data, header=False, row_names=False)

    def _parse_output(self, output_filename):
        with open(output_filename, 'r') as f:
            all_lines = f.readlines()
            rows, cols = None, None
            biclusters = []

            for i, line in enumerate(all_lines):
                if i % 2 == 0:
                    rows = self._convert(line.split())
                else:
                    cols = self._convert(line.split())
                    biclusters.append(Bicluster(rows, cols))

        return Biclustering(biclusters)

    def _convert(self, bit_array):
        return [int(i) for i, n in enumerate(bit_array) if int(n)]
