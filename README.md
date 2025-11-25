# 🌳 RAPTOR RAG LangChain

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A LangChain-integrated implementation of **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) for hierarchical document retrieval.

Based on the original [RAPTOR paper](https://arxiv.org/abs/2401.18059) and [implementation](https://github.com/parthsarthi03/raptor).

## 📋 Overview

Unlike traditional chunking methods, RAPTOR builds a **hierarchical tree structure** of documents:
- **Leaf Nodes**: Original document chunks
- **Summarized Nodes**: Clustered and summarized intermediate nodes
- **Root Nodes**: High-level thematic summaries

### Key Strategy: Collapsed Tree Retrieval

Instead of traversing the tree at query time, we:
1. **Build** the full tree
2. **Flatten (Collapse)** all nodes (Leaf + Summaries) into a single list
3. **Index** them all into a VectorStore (FAISS/Chroma)
4. **Retrieve** using standard vector similarity search

This approach enables retrieval of both granular details AND high-level thematic context.

## 🏗️ Project Structure

```
raptor-rag-langchain/
├── data/                        # Data storage
│   ├── raw/                     # Original PDFs
│   └── processed/               # Processed Indices (FAISS/Chroma)
├── src/
│   ├── __init__.py
│   ├── settings.py              # Configuration via .env
│   ├── ingestion.py             # PDF Loading & Splitting
│   │
│   ├── raptor/                  # [CORE] RAPTOR Algorithm
│   │   ├── __init__.py
│   │   ├── clustering.py        # GMM + UMAP implementation
│   │   ├── summarization.py     # LangChain Chain for Summary Generation
│   │   ├── tree_builder.py      # Recursive Tree Building Logic
│   │   └── tree_structures.py   # Data Structures (Node, RaptorTree)
│   │
│   └── retrieval/               # RAG Pipeline
│       ├── __init__.py
│       ├── vector_store.py      # VectorStore Management
│       └── rag_chain.py         # Context Retrieval & QA Generation
│
├── notebooks/                   # Experiments
├── app.py                       # Streamlit UI
├── pyproject.toml               # Dependencies (uv)
├── .env.example                 # Environment variables template
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/raptor-rag-langchain.git
cd raptor-rag-langchain

# Install dependencies with uv
uv sync

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` file with your settings:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Google AI for Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
SUMMARIZATION_MODEL=gpt-4o-mini
QA_MODEL=gpt-4o-mini

# RAPTOR Hyperparameters
MAX_TOKENS=100
NUM_LAYERS=5
TOP_K=10
```

### Usage

#### 1. Build a RAPTOR Tree from PDF

```python
from src.ingestion import DocumentIngestion
from src.raptor.tree_builder import TreeBuilder
from src.retrieval.vector_store import RaptorVectorStore

# Load and chunk document
ingestion = DocumentIngestion()
chunks = ingestion.load_and_split("path/to/document.pdf")

# Build RAPTOR tree
builder = TreeBuilder()
tree = builder.build_tree(chunks)

# Index to vector store
vector_store = RaptorVectorStore()
vector_store.add_tree(tree)
vector_store.save()
```

#### 2. Query with RAG

```python
from src.retrieval.rag_chain import RaptorRAGChain
from src.retrieval.vector_store import RaptorVectorStore

# Load existing index
vector_store = RaptorVectorStore()
vector_store.load()

# Create RAG chain
rag_chain = RaptorRAGChain(vector_store)

# Ask questions
answer = rag_chain.invoke("What is the main theme of the document?")
print(answer)
```

#### 3. Run Streamlit UI

```bash
uv run streamlit run app.py
```

## 🧪 Development

### Install Dev Dependencies

```bash
uv sync --dev
```

### Run Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Linting
uv run ruff check .

# Type checking
uv run mypy src/

# Format code
uv run ruff format .
```

## 📊 Architecture

### Core Components

| Module | Description |
|--------|-------------|
| `tree_structures.py` | Node and RaptorTree data classes with LangChain Document conversion |
| `clustering.py` | UMAP dimensionality reduction + GMM soft clustering |
| `summarization.py` | LangChain-based text summarization chain |
| `tree_builder.py` | Recursive tree construction orchestrator |
| `vector_store.py` | FAISS/Chroma vector store management |
| `rag_chain.py` | Retrieval-Augmented Generation pipeline |

### Algorithm Flow

```
PDF Document
     │
     ▼
┌─────────────┐
│  Ingestion  │ ─── PyPDFLoader + RecursiveCharacterTextSplitter
└─────────────┘
     │
     ▼
┌─────────────┐
│ Leaf Nodes  │ ─── Create embeddings for each chunk
└─────────────┘
     │
     ▼
┌─────────────┐
│ Clustering  │ ─── UMAP reduction → GMM clustering
└─────────────┘
     │
     ▼
┌─────────────┐
│Summarization│ ─── LLM summarizes each cluster
└─────────────┘
     │
     ▼
┌─────────────┐
│  Repeat     │ ─── Until reaching root node(s)
└─────────────┘
     │
     ▼
┌─────────────┐
│  Collapse   │ ─── Flatten all nodes
└─────────────┘
     │
     ▼
┌─────────────┐
│VectorStore  │ ─── Index all nodes (FAISS/Chroma)
└─────────────┘
     │
     ▼
┌─────────────┐
│  RAG Chain  │ ─── Retrieve + Generate answers
└─────────────┘
```

## 📚 References

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [Original RAPTOR Implementation](https://github.com/parthsarthi03/raptor)
- [LangChain Documentation](https://python.langchain.com/)

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
