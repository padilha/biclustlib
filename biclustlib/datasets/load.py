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
    """Load and return the yeast dataset (Tavazoie et al., 2000) used in the original biclustering study
    of Cheng and Church (2000) as a pandas.DataFrame. All elements equal to -1 are missing values. This
    dataset is freely available in http://arep.med.harvard.edu/biclustering/.

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

def load_prelic():
    """Load and return a dictionary containing the yeast (Gasch et al., 2000) and arabidopsis datasets used by
    Prelic et al. (2006). Each dictionary key is a dataset name. Each dictionary value is a pandas.DataFrame.
    These datasets are freely available in http://people.ee.ethz.ch/~sop/bimax/.

    Reference
    ---------
    Prelic, A., Bleuler, S., Zimmermann, P., Wille, A., Buhlmann, P., Gruissem, W., Hennig, L., Thiele, L. &
    Zitzler, E. (2006). A systematic comparison and evaluation of biclustering methods for gene expression data.
    Bioinformatics, 22(9), 1122-1129.

    Gasch, A. P., Spellman, P. T., Kao, C. M., Carmel-Harel, O., Eisen, M. B., Storz, G., Botstein, D. & Brown,
    P. O. (2000). Genomic expression programs in the response of yeast cells to environmental changes. Molecular
    biology of the cell, 11(12), 4241-4257.
    """
    module_dir = dirname(__file__)
    datasets_dir_path = join(module_dir, 'data', 'prelic')

    return _load_benchmark(datasets_dir_path,
                           read_func=lambda f : pd.read_table(f, delim_whitespace=True, header=0, index_col=0),
                           split_func=lambda f : f.split('.')[0])

def load_yeast_benchmark():
    """Load and return a dictionary containing the collection of 17 yeast datasets proposed as a benchmark by
    Jaskowiak et al. (2013). Each dictionary key is a dataset name. Each dictionary value is a pandas.DataFrame.
    This collection is freely available in http://lapad-web.icmc.usp.br/repositories/ieee-tcbb-2013/index.html.

    Reference
    ---------
    Jaskowiak, P. A., Campello, R. J. G. B., & Costa, I. G. (2013). Proximity measures for clustering gene expression
    microarray data: a validation methodology and a comparative analysis. IEEE/ACM transactions on computational biology
    and bioinformatics, 10(4), 845-857.
    """
    module_dir = dirname(__file__)
    benchmark_dir_path = join(module_dir, 'data', 'yeast_benchmark')

    return _load_benchmark(benchmark_dir_path,
                           read_func=lambda f : pd.read_csv(f, delim_whitespace=True, header=0, index_col=1),
                           split_func=lambda f : f.split('.')[0])

def load_cancer_benchmark():
    """Load and return a dictionary containing the collection of 35 cancer datasets proposed as a benchmark by
    de Souto et al. (2008). Each dictionary key is a dataset name. Each dictionary value is a pandas.DataFrame.

    Reference
    ---------
    de Souto, M. C., Costa, I. G., de Araujo, D. S., Ludermir, T. B., & Schliep, A. (2008). Clustering cancer gene
    expression data: a comparative study. BMC bioinformatics, 9(1), 497.
    """
    module_dir = dirname(__file__)
    benchmark_dir_path = join(module_dir, 'data', 'cancer_benchmark')

    return _load_benchmark(benchmark_dir_path,
                           read_func=lambda f : pd.read_table(f, delim_whitespace=True, header=0, index_col=0),
                           split_func=lambda f : f.split('_')[0])

def _load_benchmark(path, read_func, split_func):
    files = listdir(path)
    return {split_func(f) : read_func(join(path, f)) for f in files}
