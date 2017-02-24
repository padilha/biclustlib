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

from os.path import dirname, join
from os import listdir

import numpy as np
import pandas as pd

def load_yeast_tavazoie():
    """Load and return the yeast dataset used in the original biclustering study of Cheng and Church (2000)
    as a pandas.DataFrame. All elements equal to -1 are missing values. This dataset is freely available
    in http://arep.med.harvard.edu/biclustering/.

    Reference
    ---------
    Cheng, Y., & Church, G. M. (2000). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    Tavazoie, S., Hughes, J. D., Campbell, M. J., Cho, R. J., & Church, G. M. (1999). Systematic determination of genetic
    network architecture. Nature genetics, 22(3), 281-285.
    """
    module_dir = dirname(__file__)
    data = np.loadtxt(join(module_dir, 'data', 'yeast_tavazoie', 'yeast_tavazoie.txt'), dtype=np.double)
    genes = np.loadtxt(join(module_dir, 'data', 'yeast_tavazoie', 'genes_yeast_tavazoie.txt'), dtype=np.character)
    return pd.DataFrame(data, index=genes)

def load_yeast_benchmark():
    """Load and return a dictionary containing the collection of 17 yeast datasets proposed as a benchmark by
    Jaskowiak et al. (2013). Each dictionary key is the dataset name. Each dictionary value is a pandas.DataFrame.
    This collection is freely available in http://lapad-web.icmc.usp.br/repositories/ieee-tcbb-2013/index.html.

    Reference
    ---------
    Jaskowiak, P. A., Campello, R. J. G. B., & Costa, I. G. (2013). Proximity measures for clustering gene expression
    microarray data: a validation methodology and a comparative analysis. IEEE/ACM transactions on computational biology
    and bioinformatics, 10(4), 845-857.
    """
    module_dir = dirname(__file__)
    benchmark_dir_path = join(module_dir, 'data', 'yeast_benchmark')
    files = listdir(benchmark_dir_path)
    benchmark = {}

    for f in files:
        dataframe = pd.read_csv(join(benchmark_dir_path, f), delim_whitespace=True, index_col=[0, 1])
        name = f.split('.').pop(0)
        benchmark[name] = dataframe

    return benchmark
