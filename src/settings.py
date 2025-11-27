"""RAPTOR RAG LangChain - Configuration Settings.

This module provides centralized configuration management using pydantic-settings
and python-dotenv for environment variable handling.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RaptorSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================
    # LLM API Keys
    # ===========================================
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_base_url: str = Field(default="", description="OpenAI Base URL")
    google_api_key: str = Field(default="", description="Google AI API Key")

    # ===========================================
    # Model Configuration
    # ===========================================
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    summarization_model: str = Field(
        default="gpt-5-mini",
        description="Model for summarization tasks",
    )
    qa_model: str = Field(
        default="gpt-5-mini",
        description="Model for QA tasks",
    )

    # ===========================================
    # RAPTOR Hyperparameters
    # ===========================================
    max_tokens: int = Field(
        default=100,
        ge=1,
        description="Maximum tokens per chunk",
    )
    num_layers: int = Field(
        default=5,
        ge=1,
        description="Number of tree layers",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Top-k for retrieval",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity threshold",
    )
    selection_mode: Literal["top_k", "threshold"] = Field(
        default="top_k",
        description="Selection mode for retrieval",
    )
    summarization_length: int = Field(
        default=200,
        ge=1,
        description="Maximum tokens for summary generation",
    )

    # ===========================================
    # Clustering Configuration
    # ===========================================
    reduction_dimension: int = Field(
        default=10,
        ge=2,
        description="UMAP reduction dimension",
    )
    clustering_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="GMM threshold for soft clustering",
    )
    query_gmm_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Threshold for query relevance rejection based on GMM probability",
    )

    # ===========================================
    # Vector Store Configuration
    # ===========================================
    vector_store_type: Literal["faiss", "chroma"] = Field(
        default="faiss",
        description="Vector store backend",
    )
    vector_store_path: Path = Field(
        default=Path("./data/processed"),
        description="Path to store vector indices",
    )

    # ===========================================
    # Data Paths
    # ===========================================
    raw_data_path: Path = Field(
        default=Path("./data/raw"),
        description="Raw data directory",
    )
    processed_data_path: Path = Field(
        default=Path("./data/processed"),
        description="Processed data directory",
    )

    # ===========================================
    # Logging & Debug
    # ===========================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> RaptorSettings:
    """Get cached settings instance.

    Returns:
        RaptorSettings: Application settings loaded from environment.
    """
    return RaptorSettings()


# Convenience function for quick access
settings = get_settings()
