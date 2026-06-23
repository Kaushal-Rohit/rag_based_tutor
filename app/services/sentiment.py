"""
Sentiment Analysis & Dynamic-K Selection
==========================================
Extracted from the original ``llm_engine.py``.

Manages per-session conversation history, rolling TextBlob sentiment analysis,
and the three-band dynamic-k selector that adjusts retrieval depth based on
user comprehension state.
"""

from textblob import TextBlob

from app.core.logging_config import get_logger
from app.models.enums import UserState

logger = get_logger(__name__)


class ConversationManager:
    """
    Tracks conversation history and computes rolling sentiment.

    Maintains a sliding window of the last N interactions.
    Uses TextBlob polarity to classify user state as confused, neutral, or clear.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: list[dict] = []

    def analyze_sentiment(self, text: str) -> float:
        """Return polarity score: -1.0 (negative/confused) to 1.0 (positive/clear)."""
        lower_text = text.lower()
        confused_phrases = [
            "don't understand", "dont understand", "didnt understand", "didn't understand",
            "not clear", "confused", "explain", "more detail", "from basic", "from scratch",
            "step by step", "elaborate", "what is this", "hard to follow"
        ]
        
        # Base sentiment from TextBlob
        polarity = TextBlob(text).sentiment.polarity
        
        # Apply heuristic penalty for explicit confusion or requests for basic explanations
        if any(phrase in lower_text for phrase in confused_phrases):
            polarity = min(polarity - 0.5, -0.2)  # Ensure it drops below the -0.1 threshold
            
        return max(-1.0, min(1.0, polarity))

    def add_turn(self, query: str, answer: str) -> None:
        """Record a query-answer pair with its sentiment score."""
        sentiment = self.analyze_sentiment(query)
        self.history.append({
            "query": query,
            "answer": answer,
            "sentiment": sentiment,
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_average_sentiment(self, current_query: str | None = None) -> float:
        """Compute average sentiment across history and optionally the current query."""
        scores = [turn["sentiment"] for turn in self.history]
        if current_query:
            scores.append(self.analyze_sentiment(current_query))
        return sum(scores) / len(scores) if scores else 0.0

    def get_formatted_history(self) -> str:
        """Format conversation history as a string for LLM prompt injection."""
        if not self.history:
            return "No previous conversation history."
        lines = []
        for i, turn in enumerate(self.history):
            lines.append(f"User (Turn {i + 1}): {turn['query']}")
            lines.append(f"AI (Turn {i + 1}): {turn['answer']}\n")
        return "\n".join(lines).strip()


class DynamicKSelector:
    """
    Maps sentiment polarity to retrieval depth and system instruction.

    Thresholds (preserved exactly from original implementation):
      - polarity < -0.1  → confused → k = base_k + 3
      - -0.1 ≤ polarity ≤ 0.3 → neutral → k = base_k
      - polarity > 0.3  → clear → k = max(2, base_k - 2)
    """

    CONFUSED_THRESHOLD = -0.1
    CLEAR_THRESHOLD = 0.3

    SYSTEM_INSTRUCTIONS = {
        UserState.CONFUSED: (
            "You are a highly patient and detailed tutor. The user seems confused or is struggling "
            "to understand. Provide a very detailed, step-by-step explanation. Break down complex "
            "concepts simply. Use the provided context to answer accurately."
        ),
        UserState.NEUTRAL: (
            "You are a helpful tutor. Use the provided context to answer the user's question clearly. "
            "If the context doesn't contain the answer, say you don't know based on the book."
        ),
        UserState.CLEAR: (
            "You are an expert tutor. The user is understanding the material well. "
            "Provide highly precise, accurate, and concise answers without unnecessary fluff. "
            "Use the provided context to answer."
        ),
    }

    def __init__(self, base_k: int = 5):
        self.base_k = base_k

    def select(self, avg_sentiment: float) -> tuple[int, UserState, str]:
        """
        Determine k, user state, and system instruction from sentiment.

        Returns:
            tuple of (k, UserState, system_instruction_text)
        """
        if avg_sentiment < self.CONFUSED_THRESHOLD:
            k = self.base_k + 3
            state = UserState.CONFUSED
        elif avg_sentiment > self.CLEAR_THRESHOLD:
            k = max(2, self.base_k - 2)
            state = UserState.CLEAR
        else:
            k = self.base_k
            state = UserState.NEUTRAL

        instruction = self.SYSTEM_INSTRUCTIONS[state]
        return k, state, instruction


class SessionManager:
    """
    Manages per-session ConversationManager instances.

    Keyed by session_id so multiple users / browser tabs maintain
    independent sentiment histories.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self._sessions: dict[str, ConversationManager] = {}

    def get_or_create(self, session_id: str) -> ConversationManager:
        """Get existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationManager(
                max_history=self.max_history
            )
            logger.info(
                f"Created new session: {session_id}",
                extra={"stream": "app"},
            )
        return self._sessions[session_id]

    @property
    def active_count(self) -> int:
        return len(self._sessions)
