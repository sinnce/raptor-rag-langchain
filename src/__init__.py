"""RAPTOR RAG LangChain.

A LangChain-integrated implementation of RAPTOR (Recursive Abstractive Processing
for Tree-Organized Retrieval) for hierarchical document retrieval.
"""

from src.settings import get_settings, settings

__version__ = "0.1.0"
__all__ = ["settings", "get_settings", "__version__"]
