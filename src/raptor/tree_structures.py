"""RAPTOR Tree Structures.

This module defines the core data structures for the RAPTOR tree:
- Node: Represents a single node in the hierarchical tree
- RaptorTree: Represents the entire tree structure

Nodes can be converted to LangChain Document objects for integration
with LangChain's retrieval pipeline.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document


@dataclass
class Node:
    """Represents a node in the RAPTOR hierarchical tree.

    Attributes:
        text: The text content of the node.
        index: Unique index of the node in the tree.
        level: The layer level in the tree (0 = leaf nodes).
        children: Set of child node indices.
        embeddings: Dictionary of embeddings keyed by model name.
        node_id: Unique identifier for the node.
        metadata: Additional metadata for the node.
    """

    text: str
    index: int
    level: int = 0
    children: set[int] = field(default_factory=set)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make Node hashable by its index."""
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        """Check equality based on index."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def token_count(self) -> int:
        """Get approximate token count based on text length."""
        # Rough approximation: ~4 characters per token
        return len(self.text) // 4

    def to_document(self) -> Document:
        """Convert Node to LangChain Document.

        This method enables seamless integration with LangChain's
        VectorStore and Retriever components.

        Returns:
            Document: A LangChain Document with node metadata preserved.
        """
        metadata = {
            "node_index": self.index,
            "node_id": self.node_id,
            "level": self.level,
            "is_leaf": self.is_leaf,
            "children_ids": list(self.children),
            **self.metadata,
        }

        return Document(
            page_content=self.text,
            metadata=metadata,
        )

    @classmethod
    def from_document(
        cls,
        document: Document,
        index: int,
        level: int = 0,
        embeddings: dict[str, list[float]] | None = None,
    ) -> "Node":
        """Create a Node from a LangChain Document.

        Args:
            document: The source LangChain Document.
            index: Unique index for the node.
            level: Tree level (0 for leaf nodes).
            embeddings: Pre-computed embeddings dictionary.

        Returns:
            Node: A new Node instance.
        """
        return cls(
            text=document.page_content,
            index=index,
            level=level,
            embeddings=embeddings or {},
            metadata=document.metadata,
        )

    def add_embedding(self, model_name: str, embedding: list[float]) -> None:
        """Add an embedding for a specific model.

        Args:
            model_name: Name of the embedding model.
            embedding: The embedding vector.
        """
        self.embeddings[model_name] = embedding

    def get_embedding(self, model_name: str) -> list[float] | None:
        """Get embedding for a specific model.

        Args:
            model_name: Name of the embedding model.

        Returns:
            The embedding vector or None if not found.
        """
        return self.embeddings.get(model_name)


@dataclass
class RaptorTree:
    """Represents the complete RAPTOR hierarchical tree structure.

    Attributes:
        all_nodes: Dictionary mapping node indices to Node objects.
        root_nodes: Dictionary of root-level nodes.
        leaf_nodes: Dictionary of leaf-level (original chunk) nodes.
        num_layers: Total number of layers in the tree.
        layer_to_nodes: Mapping from layer number to list of nodes.
    """

    all_nodes: dict[int, Node] = field(default_factory=dict)
    root_nodes: dict[int, Node] = field(default_factory=dict)
    leaf_nodes: dict[int, Node] = field(default_factory=dict)
    num_layers: int = 0
    layer_to_nodes: dict[int, list[Node]] = field(default_factory=dict)

    # Models for query relevance checking (Global UMAP + GMM)
    umap_model: Any = None
    gmm_model: Any = None

    @property
    def total_nodes(self) -> int:
        """Get total number of nodes in the tree."""
        return len(self.all_nodes)

    @property
    def summary_nodes(self) -> dict[int, Node]:
        """Get all non-leaf (summary) nodes."""
        return {idx: node for idx, node in self.all_nodes.items() if not node.is_leaf}

    def get_node(self, index: int) -> Node | None:
        """Get a node by its index.

        Args:
            index: The node index.

        Returns:
            The Node or None if not found.
        """
        return self.all_nodes.get(index)

    def get_nodes_at_level(self, level: int) -> list[Node]:
        """Get all nodes at a specific level.

        Args:
            level: The tree level (0 = leaf nodes).

        Returns:
            List of nodes at the specified level.
        """
        return self.layer_to_nodes.get(level, [])

    def collapse(self) -> list[Node]:
        """Collapse the tree into a flat list of all nodes.

        This is the key operation for Collapsed Tree Retrieval:
        returns all nodes (leaves + summaries) for indexing.

        Returns:
            List of all nodes sorted by index.
        """
        return [self.all_nodes[idx] for idx in sorted(self.all_nodes.keys())]

    def to_documents(self) -> list[Document]:
        """Convert all nodes to LangChain Documents.

        Returns:
            List of Documents representing all tree nodes.
        """
        return [node.to_document() for node in self.collapse()]

    def add_node(self, node: Node) -> None:
        """Add a node to the tree.

        Args:
            node: The node to add.
        """
        self.all_nodes[node.index] = node

        # Update layer mapping
        if node.level not in self.layer_to_nodes:
            self.layer_to_nodes[node.level] = []
        self.layer_to_nodes[node.level].append(node)

        # Update leaf/root tracking
        if node.is_leaf:
            self.leaf_nodes[node.index] = node

        # Update layer count
        self.num_layers = max(self.num_layers, node.level + 1)

    def set_root_nodes(self, root_nodes: dict[int, Node]) -> None:
        """Set the root nodes of the tree.

        Args:
            root_nodes: Dictionary of root nodes.
        """
        self.root_nodes = root_nodes

    def get_children(self, node: Node) -> list[Node]:
        """Get all child nodes of a given node.

        Args:
            node: The parent node.

        Returns:
            List of child Node objects.
        """
        return [
            self.all_nodes[child_idx] for child_idx in node.children if child_idx in self.all_nodes
        ]

    def get_text_from_nodes(self, nodes: list[Node]) -> str:
        """Concatenate text from multiple nodes.

        Args:
            nodes: List of nodes.

        Returns:
            Concatenated text with newlines between nodes.
        """
        return "\n\n".join(node.text for node in nodes)

    def __repr__(self) -> str:
        """String representation of the tree."""
        return (
            f"RaptorTree(total_nodes={self.total_nodes}, "
            f"num_layers={self.num_layers}, "
            f"leaf_nodes={len(self.leaf_nodes)}, "
            f"summary_nodes={len(self.summary_nodes)})"
        )
