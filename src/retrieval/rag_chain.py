"""RAG Chain for RAPTOR.

This module implements the Retrieval-Augmented Generation chain
that combines the RAPTOR vector store with LLM-based question answering.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
                model=settings.qa_model,
                temperature=0,
                api_key=settings.openai_api_key or None,
            )
        else:
            self.llm = llm

        # Set up prompt
        template = prompt_template or DEFAULT_QA_PROMPT
        self.prompt = ChatPromptTemplate.from_template(template)

        # Get retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

        # Build the chain
        self.chain = self._build_chain()

        logger.info(
            f"Initialized RaptorRAGChain with top_k={self.top_k}, " f"model={settings.qa_model}"
        )

    def _format_docs(self, docs: list) -> str:
        """Format retrieved documents into a context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
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

        answer = self.chain.invoke(question, **kwargs)

        logger.debug(f"Generated answer of length {len(answer)}")
        return answer

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
        return await self.chain.ainvoke(question, **kwargs)

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
        model=model_name or settings.qa_model,
        temperature=temperature,
        api_key=settings.openai_api_key or None,
    )

    return RaptorRAGChain(
        vector_store=vector_store,
        llm=llm,
        top_k=top_k,
    )
