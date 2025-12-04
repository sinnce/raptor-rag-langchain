"""RAG Chain for RAPTOR.

This module implements the Retrieval-Augmented Generation chain
that combines the RAPTOR vector store with LLM-based question answering.
"""

import logging
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.retrieval.vector_store import RaptorVectorStore
from src.settings import settings


logger = logging.getLogger(__name__)

# Default QA prompt template
DEFAULT_QA_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Use the following pieces of context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


class RaptorRAGChain:
    """RAG chain for RAPTOR-based question answering.

    This class combines the RAPTOR vector store retriever with
    an LLM to perform retrieval-augmented generation.
    """

    def __init__(
        self,
        vector_store: RaptorVectorStore,
        llm: BaseChatModel | None = None,
        prompt_template: str | None = None,
        top_k: int | None = None,
    ) -> None:
        """Initialize the RAG chain.

        Args:
            vector_store: RAPTOR vector store for retrieval.
            llm: LangChain chat model for generation.
            prompt_template: Custom prompt template.
            top_k: Number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.top_k = top_k or settings.top_k

        # Initialize LLM
        if llm is None:
            self.llm = ChatOpenAI(
                base_url=settings.openai_base_url or None,
                model=settings.qa_model,
                temperature=0,
                api_key=settings.openai_api_key or None,
            )
        else:
            self.llm = llm

        # Set up prompt
        template = prompt_template or DEFAULT_QA_PROMPT
        self.prompt = ChatPromptTemplate.from_template(template)

        # Load relevance models if available
        self.umap_model, self.gmm_model = self.vector_store.load_models(
            self.vector_store.persist_directory
        )

        # Get retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

        # Build the chain
        self.chain = self._build_chain()

        logger.info(
            f"Initialized RaptorRAGChain with top_k={self.top_k}, " f"model={settings.qa_model}"
        )

    def check_query_relevance(self, query: str) -> bool:
        """Check if the query is relevant to the database using GMM models.

        Args:
            query: The query string.

        Returns:
            True if relevant, False otherwise.
        """
        if self.umap_model is None or self.gmm_model is None:
            # If no models are loaded, assume relevant (backward compatibility)
            return True

        if settings.query_gmm_threshold <= 0:
            return True

        try:
            # Embed the query
            query_embedding = self.vector_store.embedding_model.embed_query(query)
            query_embedding = np.array([query_embedding])

            # Reduce dimensionality
            reduced_embedding = self.umap_model.transform(query_embedding)

            # Predict probabilities
            probs = self.gmm_model.predict_proba(reduced_embedding)[0]

            # Check if any cluster probability exceeds the threshold
            max_prob = max(probs)
            logger.debug(f"Query max probability in GMM clusters: {max_prob:.4f}")

            return max_prob >= settings.query_gmm_threshold

        except Exception as e:
            logger.warning(f"Error checking query relevance: {e}")
            # On error, fallback to assuming relevant
            return True

    def _format_docs(self, docs: list) -> str:
        """Format retrieved documents into a context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self) -> Runnable:
        """Build the RAG chain.

        Returns:
            LangChain runnable chain.
        """
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def invoke(
        self,
        question: str,
        **kwargs: Any,
    ) -> str:
        """Answer a question using RAG.

        Args:
            question: The question to answer.
            **kwargs: Additional arguments for the chain.

        Returns:
            Generated answer string.
        """
        logger.debug(f"Processing question: {question}")

        if not self.check_query_relevance(question):
            logger.info("Query rejected due to low relevance probability.")
            return "I don't have enough information to answer this question."

        answer = self.chain.invoke(question, **kwargs)

        logger.debug(f"Generated answer of length {len(answer)}")
        return str(answer)

    async def ainvoke(
        self,
        question: str,
        **kwargs: Any,
    ) -> str:
        """Async version of invoke.

        Args:
            question: The question to answer.
            **kwargs: Additional arguments for the chain.

        Returns:
            Generated answer string.
        """
        # Note: check_query_relevance is synchronous but fast enough.
        # If strict async is needed, we should wrap it.
        if not self.check_query_relevance(question):
            logger.info("Query rejected due to low relevance probability.")
            return "I don't have enough information to answer this question."

        answer = await self.chain.ainvoke(question, **kwargs)
        return str(answer)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list:
        """Retrieve relevant documents without generation.

        Args:
            query: Query string.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved documents.
        """
        k = k or self.top_k
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple]:
        """Retrieve documents with similarity scores.

        Args:
            query: Query string.
            k: Number of documents to retrieve.

        Returns:
            List of (document, score) tuples.
        """
        k = k or self.top_k
        return self.vector_store.similarity_search_with_score(query, k=k)

    def __call__(
        self,
        question: str,
        **kwargs: Any,
    ) -> str:
        """Allow calling the chain directly.

        Args:
            question: The question to answer.
            **kwargs: Additional arguments.

        Returns:
            Generated answer string.
        """
        return self.invoke(question, **kwargs)


def create_rag_chain(
    vector_store: RaptorVectorStore,
    model_name: str | None = None,
    temperature: float = 0,
    top_k: int | None = None,
) -> RaptorRAGChain:
    """Factory function to create a RAG chain.

    Args:
        vector_store: RAPTOR vector store.
        model_name: LLM model name.
        temperature: Model temperature.
        top_k: Number of documents to retrieve.

    Returns:
        Configured RaptorRAGChain instance.
    """
    llm = ChatOpenAI(
        base_url=settings.openai_base_url or None,
        model=model_name or settings.qa_model,
        temperature=temperature,
        api_key=settings.openai_api_key or None,
    )

    return RaptorRAGChain(
        vector_store=vector_store,
        llm=llm,
        top_k=top_k,
    )
