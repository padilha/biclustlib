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

from ._base import BaseBiclusteringAlgorithm
from ..models import Bicluster, Biclustering
from sklearn.cluster import k_means
from sklearn.utils.validation import check_array

import numpy as np

class Plaid(BaseBiclusteringAlgorithm):
    """Plaid biclustering algorithm.

    This algorithm fits the plaid model using a binary least squares approach.

    Reference
    ----------
    Turner, H., Bailey, T., & Krzanowski, W. (2005). Improved biclustering of microarray data demonstrated
    through systematic performance tests. Computational statistics & data analysis, 48(2), 235-254.

    Parameters
    ----------
    num_biclusters : int, default: 10
        Number of biclusters to be found.

    fit_background_layer : bool, default: True
        If True fits a background layer which represents common effects of all elements of the data matrix.

    row_prunning_threshold : float, default: 0.7
        Threshold for row prunning.

    col_prunning_threshold : float, default: 0.7
        Threshold for column prunning.

    significance_tests : int, default: 0
        Number of significance tests to be performed for a layer.

    back_fitting_steps : int, default: 1
        Number of back fitting steps.

    initialization_iterations : int, default: 6
        Number of k-means runs to initialize a new layer.

    iterations_per_layer : int, default: 10
        Number of prunning iterations per layer.
    """

    def __init__(self, num_biclusters=10, fit_background_layer=True, row_prunning_threshold=0.7,
                 col_prunning_threshold=0.7, significance_tests=0, back_fitting_steps=1,
                 initialization_iterations=6, iterations_per_layer=10):
        self.num_biclusters = num_biclusters
        self.fit_background_layer = fit_background_layer
        self.row_prunning_threshold = row_prunning_threshold
        self.col_prunning_threshold = col_prunning_threshold
        self.significance_tests = significance_tests
        self.back_fitting_steps = back_fitting_steps
        self.initialization_iterations = initialization_iterations
        self.iterations_per_layer = iterations_per_layer

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """
        residuals = check_array(data, dtype=np.double, copy=True)
        self._validate_parameters()

        num_rows, num_cols = residuals.shape
        biclusters, layers = [], []

        if self.fit_background_layer:
            background_layer = self._create_layer(residuals)
            layers.append(background_layer)
            residuals -= background_layer
            biclusters.append(Bicluster(np.arange(num_rows), np.arange(num_cols)))

        for i in range(self.num_biclusters):
            rows, cols, bicluster_layer = self._fit_layer(residuals)

            if len(rows) == 0 or len(cols) == 0 or not self._is_significant(residuals, bicluster_layer):
                break

            residuals[rows[:, np.newaxis], cols] -= bicluster_layer

            layers.append(bicluster_layer)
            biclusters.append(Bicluster(rows, cols))

            self._back_fitting(residuals, layers, biclusters)

        biclustering = Biclustering(biclusters)

        if self.fit_background_layer:
            biclusters.pop(0)

        return biclustering

    def _generate_initial_bicluster(self, residuals):
        """Generates a initial bicluster based on the results of several K-means runs over the matrix of residuals."""
        rows = self._kmeans_initialization(residuals)
        cols = self._kmeans_initialization(residuals.T)
        return rows, cols

    def _create_layer(self, residuals):
        """Creates a new layer."""
        mean = np.mean(residuals)
        row_effects = np.mean(residuals, axis=1) - mean
        col_effects = np.mean(residuals, axis=0) - mean
        return mean + row_effects[:, np.newaxis] + col_effects

    def _kmeans_initialization(self, residuals):
        """Computes k-means with k = 2 to find the initial components (rows or columns) of a new layer/bicluster."""
        _, labels, _ = k_means(residuals, n_clusters=2, n_init=self.initialization_iterations, init='random', n_jobs=1)
        count0, count1 = np.bincount(labels)

        if count0 <= count1:
            return np.where(labels == 0)[0]

        return np.where(labels == 1)[0]

    def _fit_layer(self, residuals):
        """Fits a new layer."""
        rows, cols = self._generate_initial_bicluster(residuals)
        layer = self._create_layer(residuals[rows[:, np.newaxis], cols])

        for i in range(self.iterations_per_layer):
            rows_old = np.copy(rows)
            rows = self._prune(residuals, layer, rows, cols, self.row_prunning_threshold)
            cols = self._prune(residuals.T, layer.T, cols, rows_old, self.col_prunning_threshold)

            if len(rows) == 0 or len(cols) == 0:
                break

            layer = self._create_layer(residuals[rows[:, np.newaxis], cols])

        return rows, cols, layer

    def _prune(self, residuals, layer, rows, cols, prunning_threshold):
        """Prune rows and columns from the layer/bicluster being computed."""
        res = residuals[rows[:, np.newaxis], cols]
        diff = res - layer
        sum_squared_diff = np.sum(diff * diff, axis=1)
        sum_squared_residuals = np.sum(res * res, axis=1)
        selected_indices = np.where(sum_squared_diff < (1.0 - prunning_threshold) * sum_squared_residuals)[0]
        return rows[selected_indices]

    def _is_significant(self, residuals, layer):
        """Tests a layer for significance."""
        layer_sum_of_squares = np.sum(layer * layer)
        shuffled_residuals = np.copy(residuals)

        for i in range(self.significance_tests):
            np.random.shuffle(shuffled_residuals.flat)
            _, _, test_layer = self._fit_layer(shuffled_residuals)
            test_layer_sum_of_squares = np.sum(test_layer * test_layer)

            # If the sum of squares of the layer found in the shuffled dataset is greater than the sum of
            # squares of the layer found in the original dataset, stop. The layer found is not significant.
            if test_layer_sum_of_squares >= layer_sum_of_squares:
                return False

        return True

    def _back_fitting(self, residuals, layers, biclusters):
        """Performs back fitting steps."""
        for i in range(self.back_fitting_steps):
            for j, b in zip(range(len(layers)), biclusters):
                residuals[b.rows[:, np.newaxis], b.cols] += layers[j]
                layers[j] = self._create_layer(residuals[b.rows[:, np.newaxis], b.cols])
                residuals[b.rows[:, np.newaxis], b.cols] -= layers[j]

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if not isinstance(self.fit_background_layer, bool):
            raise ValueError("fit_background_layer must be either True or False, got {}".format(self.fit_background_layer))

        if self.initialization_iterations <= 0:
            raise ValueError("initialization_iterations must be > 0, got {}".format(self.initialization_iterations))

        if self.iterations_per_layer <= 0:
            raise ValueError("iterations_per_layer must be > 0, got {}".format(self.iterations_per_layer))

        if self.significance_tests < 0:
            raise ValueError("significance_tests must be >= 0, got {}".format(self.significance_tests))

        if self.back_fitting_steps < 0:
            raise ValueError("back_fitting_steps must be >= 0, got {}".format(self.back_fitting_steps))

        if not (0.0 < self.row_prunning_threshold < 1.0):
            raise ValueError("row_prunning_threshold must be > 0.0 and < 1.0, got {}".format(self.row_prunning_threshold))

        if not (0.0 < self.col_prunning_threshold < 1.0):
            raise ValueError("col_prunning_threshold must be > 0.0 and < 1.0, got {}".format(self.col_prunning_threshold))
