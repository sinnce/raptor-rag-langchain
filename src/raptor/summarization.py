"""Summarization module for RAPTOR.

This module implements LangChain-based summarization chains
for generating summaries of clustered text nodes.
"""

import logging
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.settings import settings


logger = logging.getLogger(__name__)

# Default summarization prompt
DEFAULT_SUMMARIZATION_PROMPT = """You are an expert summarizer. Your task is to create a comprehensive summary of the following texts.

Instructions:
1. Identify the common themes and key information across all texts
2. Preserve important details, facts, and relationships
3. Create a coherent summary that captures the essential information
4. Keep the summary concise but informative

Texts to summarize:
{texts}

Provide a clear and concise summary:"""


class SummarizationChain:
    """LangChain-based summarization chain for RAPTOR.

    This class wraps a LangChain chain for generating summaries
    of grouped text nodes.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        prompt_template: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the summarization chain.

        Args:
            llm: LangChain chat model. Defaults to OpenAI.
            prompt_template: Custom prompt template. Uses default if not provided.
            max_tokens: Maximum tokens for summary output.
        """
        self.max_tokens = max_tokens or settings.summarization_length

        # Initialize LLM
        if llm is None:
            self.llm = ChatOpenAI(
                base_url=settings.openai_base_url or None,
                model=settings.summarization_model,
                temperature=0,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key or None,
            )
        else:
            self.llm = llm

        # Set up prompt template
        template = prompt_template or DEFAULT_SUMMARIZATION_PROMPT
        self.prompt = ChatPromptTemplate.from_template(template)

        # Build the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

        logger.info(
            f"Initialized SummarizationChain with model={settings.summarization_model}, "
            f"max_tokens={self.max_tokens}"
        )

    def summarize(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> str:
        """Generate a summary of multiple texts.

        Args:
            texts: List of text strings to summarize.
            **kwargs: Additional arguments passed to the chain.

        Returns:
            Summary string.
        """
        # Combine texts with separators
        combined_text = "\n\n---\n\n".join(texts)

        # Generate summary
        try:
            summary = self.chain.invoke(
                {"texts": combined_text},
                **kwargs,
            )
            logger.debug(f"Generated summary of length {len(summary)}")
            return cast(str, summary)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def summarize_nodes(
        self,
        nodes: list,  # List[Node] - avoiding circular import
    ) -> str:
        """Generate a summary of multiple nodes.

        Args:
            nodes: List of Node objects to summarize.

        Returns:
            Summary string.
        """
        texts = [node.text for node in nodes]
        return self.summarize(texts)

    async def asummarize(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> str:
        """Async version of summarize.

        Args:
            texts: List of text strings to summarize.
            **kwargs: Additional arguments passed to the chain.

        Returns:
            Summary string.
        """
        combined_text = "\n\n---\n\n".join(texts)

        try:
            summary = await self.chain.ainvoke(
                {"texts": combined_text},
                **kwargs,
            )
            return cast(str, summary)
        except Exception as e:
            logger.error(f"Error generating async summary: {e}")
            raise


def summarize_texts(
    texts: list[str],
    llm: BaseChatModel | None = None,
    max_tokens: int | None = None,
) -> str:
    """Utility function to summarize texts.

    Args:
        texts: List of text strings to summarize.
        llm: Optional custom LLM.
        max_tokens: Maximum tokens for output.

    Returns:
        Summary string.
    """
    chain = SummarizationChain(llm=llm, max_tokens=max_tokens)
    return chain.summarize(texts)


def get_summarization_chain(
    model_name: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> SummarizationChain:
    """Factory function to create a summarization chain.

    Args:
        model_name: Name of the model to use.
        temperature: Model temperature.
        max_tokens: Maximum output tokens.

    Returns:
        Configured SummarizationChain instance.
    """
    llm = ChatOpenAI(
        base_url=settings.openai_base_url or None,
        model=model_name or settings.summarization_model,
        temperature=temperature,
        max_tokens=max_tokens or settings.summarization_length,
        api_key=settings.openai_api_key or None,
    )

    return SummarizationChain(llm=llm, max_tokens=max_tokens)
