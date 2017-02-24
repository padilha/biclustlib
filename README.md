# biclustlib

biclustlib is a Python library of biclustering algorithms and evaluation measures distributed under the GPLv3 license.

## Installation

TODO

### Dependencies

See requirements.txt.

## Example of use

```python
import numpy as np

from biclustlib.algorithms import ChengChurchAlgorithm
from biclustlib.datasets import load_yeast_tavazoie

# load yeast data used in the original Cheng and Church's paper
data = load_yeast_tavazoie().values

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
cca = ChengChurchAlgorithm(num_biclusters=100, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
biclustering = cca.run(data)
print(biclustering)
```

## Citation
If you use biclustlib in a scientific publication, we would appreciate citations of our paper where this library was originally proposed.

To cite biclustlib use: Padilha, V. A. & Campello, R. J. G. B. (2017). A systematic comparative evaluation of biclustering techniques. *BMC Bioinformatics*, 18(1):55.

For TeX/LaTeX:

    @article{padilha2017,
      title={A systematic comparative evaluation of biclustering techniques},
      author={Padilha, Victor A and Campello, Ricardo J G B},
      journal={BMC Bioinformatics},
      volume={18},
      number={1},
      pages={55},
      year={2017},
      publisher={BioMed Central}
    }

## License (GPLv3)
    biclustlib: A Python library of biclustering algorithms and evaluation measures.
    Copyright (C) 2017  Victor Alexandre Padilha

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
