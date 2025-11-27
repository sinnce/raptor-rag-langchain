"""Tests for clustering module."""

import numpy as np
import pytest

from src.raptor.clustering import (
    RaptorClustering,
    get_optimal_clusters,
    gmm_cluster,
    perform_clustering,
)


class TestClusteringFunctions:
    """Test cases for clustering functions."""

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 10)

    def test_get_optimal_clusters(self, sample_embeddings: np.ndarray) -> None:
        """Test finding optimal number of clusters."""
        n_clusters = get_optimal_clusters(sample_embeddings, max_clusters=5)

        assert isinstance(n_clusters, int)
        assert 1 <= n_clusters <= 5

    def test_gmm_cluster(self, sample_embeddings: np.ndarray) -> None:
        """Test GMM clustering."""
        labels, n_clusters, gm_model = gmm_cluster(sample_embeddings, threshold=0.1)

        assert len(labels) == len(sample_embeddings)
        assert n_clusters >= 1
        assert all(isinstance(label, np.ndarray) for label in labels)
        assert gm_model is not None

    def test_perform_clustering(self, sample_embeddings: np.ndarray) -> None:
        """Test full clustering pipeline."""
        clusters = perform_clustering(
            sample_embeddings,
            dim=5,
            threshold=0.1,
        )

        assert len(clusters) == len(sample_embeddings)

    def test_clustering_edge_case_small(self) -> None:
        """Test clustering with very few samples."""
        small_embeddings = np.random.randn(3, 10)

        clusters = perform_clustering(
            small_embeddings,
            dim=2,
            threshold=0.1,
        )

        # Should handle gracefully
        assert len(clusters) == 3


class TestRaptorClustering:
    """Test cases for RaptorClustering class."""

    def test_initialization(self) -> None:
        """Test RaptorClustering initialization."""
        clustering = RaptorClustering(
            reduction_dimension=5,
            threshold=0.2,
        )

        assert clustering.reduction_dimension == 5
        assert clustering.threshold == 0.2

    def test_perform_clustering(self) -> None:
        """Test clustering via class method."""
        np.random.seed(42)
        embeddings = np.random.randn(15, 8)

        clustering = RaptorClustering(
            reduction_dimension=3,
            threshold=0.1,
        )

        result = clustering.perform_clustering(embeddings)

        assert isinstance(result, list)
        assert all(isinstance(cluster, list) for cluster in result)
