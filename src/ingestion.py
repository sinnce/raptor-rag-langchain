"""Document ingestion module.

This module handles PDF loading and text splitting for the RAPTOR pipeline.
Uses LangChain's document loaders and text splitters.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.settings import settings


logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handles document loading and chunking for RAPTOR."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int = 50,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        """Initialize the document ingestion pipeline.

        Args:
            chunk_size: Maximum tokens per chunk. Defaults to settings.max_tokens.
            chunk_overlap: Number of overlapping tokens between chunks.
            tokenizer_name: Tiktoken encoding name for token counting.
        """
        self.chunk_size = chunk_size or settings.max_tokens
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

        # Initialize text splitter with token-based length function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"Initialized DocumentIngestion with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Input text string.

        Returns:
            Number of tokens in the text.
        """
        return len(self.tokenizer.encode(text))

    def load_pdf(self, pdf_path: str | Path) -> list[Document]:
        """Load a PDF file and return raw documents.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
        return documents

    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of Document objects to split.

        Returns:
            List of chunked Document objects.
        """
        chunks = self.text_splitter.split_documents(list(documents))
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def load_and_split(self, pdf_path: str | Path) -> list[Document]:
        """Load a PDF and split it into chunks in one step.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of chunked Document objects.
        """
        documents = self.load_pdf(pdf_path)
        return self.split_documents(documents)

    def load_directory(
        self,
        directory: str | Path | None = None,
        glob_pattern: str = "*.pdf",
    ) -> list[Document]:
        """Load all PDFs from a directory.

        Args:
            directory: Path to directory. Defaults to settings.raw_data_path.
            glob_pattern: Glob pattern for PDF files.

        Returns:
            List of all chunked Document objects from all PDFs.
        """
        directory = Path(directory) if directory else settings.raw_data_path

        all_chunks: list[Document] = []
        pdf_files = list(directory.glob(glob_pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return all_chunks

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        for pdf_path in pdf_files:
            try:
                chunks = self.load_and_split(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue

        logger.info(f"Total chunks from directory: {len(all_chunks)}")
        return all_chunks


def create_chunks_from_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int = 50,
) -> list[str]:
    """Utility function to split raw text into chunks.

    Args:
        text: Raw text string to split.
        chunk_size: Maximum tokens per chunk. Defaults to settings.max_tokens.
        chunk_overlap: Number of overlapping tokens.

    Returns:
        List of text chunks as strings.
    """
    ingestion = DocumentIngestion(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create a Document from raw text
    doc = Document(page_content=text, metadata={"source": "raw_text"})
    chunks = ingestion.split_documents([doc])

    return [chunk.page_content for chunk in chunks]
