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

from ._base import ExecutableWrapper
from ...models import Bicluster, Biclustering
from os.path import dirname, join

import numpy as np

class RInClose(ExecutableWrapper):
    """RInClose algorithm
    TODO: add doc
    """

    def __init__(self, min_rows=2, min_cols=2, noise_tol=0.01, algorithm='chv'):
        super().__init__()
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.noise_tol = noise_tol
        self.algorithm = algorithm

    def _get_command(self, data, data_path, output_path):
        module_dir = dirname(__file__)

        return join(module_dir, 'bin', 'rinclose', 'RInClose') + ' ' + \
                    ' '.join((data_path,
                              self.algorithm,
                              str(self.min_rows),
                              str(self.min_cols),
                              str(self.noise_tol),
                              output_path))

    def _write_data(self, data_path, data):
        np.savetxt(data_path, data)

    def _parse_output(self, output_path):
        biclusters = []

        with open(output_path, 'r') as f:
            all_lines = f.readlines()

            for i in range(0, len(all_lines) - 1, 2):
                rows = self._line_to_array(all_lines[i])
                cols = self._line_to_array(all_lines[i + 1])
                b = Bicluster(rows, cols)
                biclusters.append(b)

        return Biclustering(biclusters)

    def _line_to_array(self, line):
        _, line_content = line.split(' = ')
        line_content = line_content[1:-3].strip() # removing \n [ ] and ; from string
        return np.array(line_content.split(), dtype=np.int) - 1 # because the output is in MATLAB style

    def _validate_parameters(self):
        pass
