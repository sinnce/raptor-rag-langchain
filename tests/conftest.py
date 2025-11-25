"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
    Once upon a time, in a land far away, there lived a young princess.
    She was kind and gentle, and everyone in the kingdom loved her.
    One day, she decided to explore the enchanted forest near the castle.
    In the forest, she met a wise old owl who told her about a hidden treasure.
    The princess embarked on a journey to find the treasure, facing many challenges.
    """


@pytest.fixture
def sample_chunks() -> list[str]:
    """Sample text chunks for testing."""
    return [
        "Once upon a time, in a land far away, there lived a young princess.",
        "She was kind and gentle, and everyone in the kingdom loved her.",
        "One day, she decided to explore the enchanted forest near the castle.",
        "In the forest, she met a wise old owl who told her about a hidden treasure.",
        "The princess embarked on a journey to find the treasure, facing many challenges.",
    ]
