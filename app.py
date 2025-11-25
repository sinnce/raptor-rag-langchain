"""RAPTOR RAG LangChain - Streamlit Application.

A web interface for the RAPTOR RAG pipeline.
"""

import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import DocumentIngestion
from src.raptor.tree_builder import TreeBuilder
from src.retrieval.rag_chain import RaptorRAGChain
from src.retrieval.vector_store import RaptorVectorStore
from src.settings import settings


# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tree_built" not in st.session_state:
        st.session_state.tree_built = False


def sidebar_config() -> None:
    """Render sidebar configuration."""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.subheader("Model Settings")
    st.sidebar.text(f"Embedding: {settings.embedding_model}")
    st.sidebar.text(f"QA Model: {settings.qa_model}")
    st.sidebar.text(f"Summarization: {settings.summarization_model}")

    st.sidebar.subheader("RAPTOR Settings")
    st.sidebar.text(f"Max Tokens: {settings.max_tokens}")
    st.sidebar.text(f"Num Layers: {settings.num_layers}")
    st.sidebar.text(f"Top-K: {settings.top_k}")

    st.sidebar.divider()

    # Load existing index
    st.sidebar.subheader("📂 Load Existing Index")
    if st.sidebar.button("Load from disk"):
        try:
            vector_store = RaptorVectorStore()
            vector_store.load()
            st.session_state.vector_store = vector_store
            st.session_state.rag_chain = RaptorRAGChain(vector_store)
            st.session_state.tree_built = True
            st.sidebar.success("Index loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading index: {e}")


def upload_and_process() -> None:
    """Handle PDF upload and processing."""
    st.subheader("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF to build the RAPTOR tree",
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path(settings.raw_data_path) / uploaded_file.name
        settings.raw_data_path.mkdir(parents=True, exist_ok=True)

        with temp_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("🌳 Build RAPTOR Tree", type="primary"):
            with st.spinner("Building RAPTOR tree... This may take a few minutes."):
                try:
                    # Step 1: Ingest document
                    st.info("Step 1/3: Loading and chunking document...")
                    ingestion = DocumentIngestion()
                    chunks = ingestion.load_and_split(temp_path)
                    st.write(f"Created {len(chunks)} chunks")

                    # Step 2: Build tree
                    st.info("Step 2/3: Building hierarchical tree...")
                    builder = TreeBuilder()
                    tree = builder.build_tree(chunks)
                    st.write(f"Built tree with {tree.total_nodes} nodes, {tree.num_layers} layers")

                    # Step 3: Create vector store
                    st.info("Step 3/3: Indexing to vector store...")
                    vector_store = RaptorVectorStore()
                    vector_store.add_tree(tree)
                    vector_store.save()

                    # Update session state
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = RaptorRAGChain(vector_store)
                    st.session_state.tree_built = True

                    st.success("✅ RAPTOR tree built and indexed successfully!")

                except Exception as e:
                    logger.error(f"Error building tree: {e}")
                    st.error(f"Error: {e}")


def chat_interface() -> None:
    """Render the chat interface."""
    st.subheader("💬 Ask Questions")

    if not st.session_state.tree_built:
        st.info("Please upload a document and build the RAPTOR tree first.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                logger.error(error_msg)


def show_retrieved_context() -> None:
    """Show retrieved context for debugging."""
    if not st.session_state.tree_built:
        return

    with st.expander("🔍 Debug: View Retrieved Context"):
        query = st.text_input("Enter a query to see retrieved context")

        if query and st.button("Retrieve"):
            try:
                results = st.session_state.rag_chain.retrieve_with_scores(query)

                for i, (doc, score) in enumerate(results):
                    st.write(f"**Result {i+1}** (Score: {score:.4f})")
                    st.write(f"Level: {doc.metadata.get('level', 'N/A')}")
                    st.write(f"Is Leaf: {doc.metadata.get('is_leaf', 'N/A')}")
                    st.text(doc.page_content[:500] + "...")
                    st.divider()
            except Exception as e:
                st.error(f"Error: {e}")


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="RAPTOR RAG",
        page_icon="🌳",
        layout="wide",
    )

    st.title("🌳 RAPTOR RAG with LangChain")
    st.caption("Recursive Abstractive Processing for Tree-Organized Retrieval")

    init_session_state()
    sidebar_config()

    # Main content
    tab1, tab2 = st.tabs(["📄 Upload & Build", "💬 Chat"])

    with tab1:
        upload_and_process()

    with tab2:
        chat_interface()
        show_retrieved_context()


if __name__ == "__main__":
    main()
