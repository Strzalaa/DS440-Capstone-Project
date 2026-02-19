"""Tests for the clustering module."""

import numpy as np
import pytest


class TestKmeansOptimalK:
    def test_returns_valid_k(self):
        """Optimal k should be within the tested range."""
        pytest.skip("Not yet implemented")

    def test_silhouette_scores_in_range(self):
        """Silhouette scores should be between -1 and 1."""
        pytest.skip("Not yet implemented")


class TestCharacterizeClusters:
    def test_one_row_per_cluster(self):
        """Output should have one summary row per unique cluster label."""
        pytest.skip("Not yet implemented")
