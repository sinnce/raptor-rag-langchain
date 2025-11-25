"""Clustering module for RAPTOR.

This module implements the GMM (Gaussian Mixture Model) + UMAP clustering
algorithm used in RAPTOR for grouping similar text embeddings.

Based on the original RAPTOR implementation:
https://github.com/parthsarthi03/raptor/blob/main/raptor/cluster_utils.py
"""

import logging
import random
from abc import ABC, abstractmethod

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

from src.settings import settings

logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce embeddings dimensionality globally using UMAP.
    
    Args:
        embeddings: Array of embedding vectors.
        dim: Target dimensionality.
        n_neighbors: Number of neighbors for UMAP. Defaults to sqrt(n-1).
        metric: Distance metric for UMAP.
        
    Returns:
        Reduced embedding array.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED,
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce embeddings dimensionality locally using UMAP.
    
    Args:
        embeddings: Array of embedding vectors.
        dim: Target dimensionality.
        n_neighbors: Number of neighbors for UMAP.
        metric: Distance metric for UMAP.
        
    Returns:
        Reduced embedding array.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED,
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED,
) -> int:
    """Find optimal number of clusters using BIC.
    
    Uses Bayesian Information Criterion (BIC) to determine
    the optimal number of Gaussian mixture components.
    
    Args:
        embeddings: Array of embedding vectors.
        max_clusters: Maximum number of clusters to try.
        random_state: Random seed for reproducibility.
        
    Returns:
        Optimal number of clusters.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters_range = np.arange(1, max_clusters)
    bics = []
    
    for n in n_clusters_range:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    
    optimal_clusters = n_clusters_range[np.argmin(bics)]
    return int(optimal_clusters)


def gmm_cluster(
    embeddings: np.ndarray,
    threshold: float,
    random_state: int = RANDOM_SEED,
) -> tuple[list[np.ndarray], int]:
    """Perform GMM clustering with soft assignment.
    
    Uses Gaussian Mixture Model for soft clustering where
    each point can belong to multiple clusters based on
    probability threshold.
    
    Args:
        embeddings: Array of embedding vectors.
        threshold: Probability threshold for cluster assignment.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (cluster labels per point, number of clusters).
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Perform hierarchical clustering with UMAP + GMM.
    
    This is the main clustering function that:
    1. Reduces dimensionality globally using UMAP
    2. Performs global GMM clustering
    3. For each global cluster, performs local UMAP + GMM
    
    Args:
        embeddings: Array of embedding vectors.
        dim: Target dimensionality for UMAP reduction.
        threshold: Probability threshold for GMM assignment.
        verbose: Enable verbose logging.
        
    Returns:
        List of cluster assignments for each embedding.
    """
    # Handle edge cases
    if len(embeddings) <= dim + 1:
        # Not enough samples for UMAP reduction
        return [np.array([0]) for _ in range(len(embeddings))]
    
    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, 
        min(dim, len(embeddings) - 2)
    )
    
    # Global clustering
    global_clusters, n_global_clusters = gmm_cluster(
        reduced_embeddings_global, 
        threshold
    )
    
    if verbose:
        logger.info(f"Global Clusters: {n_global_clusters}")
    
    all_local_clusters: list[np.ndarray] = [
        np.array([]) for _ in range(len(embeddings))
    ]
    total_clusters = 0
    
    # Process each global cluster
    for i in range(n_global_clusters):
        # Get embeddings belonging to this global cluster
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_mask]
        
        if verbose:
            logger.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        
        if len(global_cluster_embeddings_) == 0:
            continue
        
        # Local clustering within global cluster
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, 
                dim
            )
            local_clusters, n_local_clusters = gmm_cluster(
                reduced_embeddings_local, 
                threshold
            )
        
        if verbose:
            logger.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")
        
        # Map local clusters back to original indices
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], 
                    j + total_clusters
                )
        
        total_clusters += n_local_clusters
    
    if verbose:
        logger.info(f"Total Clusters: {total_clusters}")
    
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def perform_clustering(
        self,
        embeddings: np.ndarray,
        **kwargs,
    ) -> list[list[int]]:
        """Perform clustering on embeddings.
        
        Args:
            embeddings: Array of embedding vectors.
            **kwargs: Additional algorithm-specific parameters.
            
        Returns:
            List of cluster assignments (indices) for each group.
        """
        pass


class RaptorClustering(ClusteringAlgorithm):
    """RAPTOR clustering implementation using UMAP + GMM.
    
    This class provides a high-level interface for the RAPTOR
    clustering algorithm that handles nodes directly.
    """
    
    def __init__(
        self,
        reduction_dimension: int | None = None,
        threshold: float | None = None,
        max_length_in_cluster: int = 3500,
        verbose: bool = False,
    ) -> None:
        """Initialize RAPTOR clustering.
        
        Args:
            reduction_dimension: UMAP target dimension.
            threshold: GMM probability threshold.
            max_length_in_cluster: Max tokens per cluster before re-clustering.
            verbose: Enable verbose logging.
        """
        self.reduction_dimension = (
            reduction_dimension or settings.reduction_dimension
        )
        self.threshold = threshold or settings.clustering_threshold
        self.max_length_in_cluster = max_length_in_cluster
        self.verbose = verbose

    def perform_clustering(
        self,
        embeddings: np.ndarray,
        **kwargs,
    ) -> list[list[int]]:
        """Perform clustering on embeddings.
        
        Args:
            embeddings: Array of embedding vectors.
            **kwargs: Additional parameters.
            
        Returns:
            List of lists, where each inner list contains indices
            of embeddings belonging to that cluster.
        """
        clusters = perform_clustering(
            embeddings,
            dim=self.reduction_dimension,
            threshold=self.threshold,
            verbose=self.verbose,
        )
        
        # Convert from per-embedding clusters to per-cluster indices
        cluster_to_indices: dict[int, list[int]] = {}
        
        for idx, cluster_assignments in enumerate(clusters):
            for cluster_id in cluster_assignments:
                cluster_id = int(cluster_id)
                if cluster_id not in cluster_to_indices:
                    cluster_to_indices[cluster_id] = []
                cluster_to_indices[cluster_id].append(idx)
        
        return list(cluster_to_indices.values())

    def cluster_nodes(
        self,
        nodes: list,  # List[Node] - avoiding circular import
        embedding_model_name: str,
    ) -> list[list]:  # List[List[Node]]
        """Cluster nodes based on their embeddings.
        
        Args:
            nodes: List of Node objects with embeddings.
            embedding_model_name: Name of the embedding model to use.
            
        Returns:
            List of node clusters (each cluster is a list of nodes).
        """
        if len(nodes) <= 1:
            return [nodes] if nodes else []
        
        # Extract embeddings from nodes
        embeddings = np.array([
            node.embeddings[embedding_model_name] 
            for node in nodes
        ])
        
        # Perform clustering
        cluster_indices = self.perform_clustering(embeddings)
        
        # Map indices back to nodes
        node_clusters = [
            [nodes[idx] for idx in cluster]
            for cluster in cluster_indices
        ]
        
        return node_clusters
