"""
Enum Definitions
================
Strict enum types for metadata field validation.
Maps to the known values in the NCERT corpus.
"""

from enum import Enum


class Subject(str, Enum):
    """Valid subject values in the corpus metadata."""
    PHYSICS = "Physics"
    CHEMISTRY = "Chemistry"
    ENGLISH = "English"
    SCIENCE = "Science"


class ClassLevel(str, Enum):
    """Valid class levels in the corpus metadata."""
    CLASS_9 = "9"
    CLASS_11 = "11"
    CLASS_12 = "12"


class ContentType(str, Enum):
    """Valid content_type values in the corpus metadata."""
    THEORY = "theory"
    DEFINITION = "definition"
    FORMULA = "formula"
    EXERCISE = "exercise"
    QA_PAIR = "qa_pair"
    EXAMPLE = "example"
    SUMMARY = "summary"


class RetrievalBackend(str, Enum):
    """Supported retrieval backends."""
    CHROMA = "chroma"
    FAISS = "faiss"
    BOTH = "both"


class UserState(str, Enum):
    """Sentiment-derived user comprehension states."""
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    CLEAR = "clear"
