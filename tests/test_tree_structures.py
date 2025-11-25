"""Tests for RAPTOR tree structures."""

import pytest

from src.raptor.tree_structures import Node, RaptorTree


class TestNode:
    """Test cases for Node class."""

    def test_node_creation(self) -> None:
        """Test basic node creation."""
        node = Node(
            text="Test text",
            index=0,
            level=0,
        )
        
        assert node.text == "Test text"
        assert node.index == 0
        assert node.level == 0
        assert node.is_leaf is True
        assert len(node.children) == 0

    def test_node_with_children(self) -> None:
        """Test node with children."""
        node = Node(
            text="Parent node",
            index=5,
            level=1,
            children={0, 1, 2},
        )
        
        assert node.is_leaf is False
        assert len(node.children) == 3
        assert 0 in node.children

    def test_node_to_document(self) -> None:
        """Test conversion to LangChain Document."""
        node = Node(
            text="Test content",
            index=0,
            level=1,
            children={1, 2},
        )
        
        doc = node.to_document()
        
        assert doc.page_content == "Test content"
        assert doc.metadata["node_index"] == 0
        assert doc.metadata["level"] == 1
        assert doc.metadata["is_leaf"] is False

    def test_node_embedding(self) -> None:
        """Test adding and retrieving embeddings."""
        node = Node(text="Test", index=0)
        
        embedding = [0.1, 0.2, 0.3]
        node.add_embedding("test_model", embedding)
        
        assert node.get_embedding("test_model") == embedding
        assert node.get_embedding("nonexistent") is None


class TestRaptorTree:
    """Test cases for RaptorTree class."""

    def test_tree_creation(self) -> None:
        """Test basic tree creation."""
        tree = RaptorTree()
        
        assert tree.total_nodes == 0
        assert tree.num_layers == 0

    def test_add_node(self) -> None:
        """Test adding nodes to tree."""
        tree = RaptorTree()
        
        node = Node(text="Leaf node", index=0, level=0)
        tree.add_node(node)
        
        assert tree.total_nodes == 1
        assert tree.num_layers == 1
        assert 0 in tree.leaf_nodes

    def test_collapse_tree(self) -> None:
        """Test collapsing tree to flat list."""
        tree = RaptorTree()
        
        # Add leaf nodes
        for i in range(3):
            tree.add_node(Node(text=f"Leaf {i}", index=i, level=0))
        
        # Add summary node
        tree.add_node(Node(
            text="Summary",
            index=3,
            level=1,
            children={0, 1, 2},
        ))
        
        collapsed = tree.collapse()
        
        assert len(collapsed) == 4
        assert all(isinstance(n, Node) for n in collapsed)

    def test_to_documents(self) -> None:
        """Test converting tree to Documents."""
        tree = RaptorTree()
        tree.add_node(Node(text="Test", index=0, level=0))
        tree.add_node(Node(text="Test 2", index=1, level=0))
        
        docs = tree.to_documents()
        
        assert len(docs) == 2
        assert all(hasattr(d, "page_content") for d in docs)
