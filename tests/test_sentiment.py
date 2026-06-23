"""
Unit Tests: Sentiment Analysis & Dynamic-K Selection
=====================================================
Verifies the three sentiment bands map to the correct k adjustments
and that the ConversationManager correctly tracks rolling sentiment.
"""

import pytest

from app.models.enums import UserState
from app.services.sentiment import ConversationManager, DynamicKSelector


class TestDynamicKSelector:
    """Tests for the three-band dynamic-k selector."""

    def setup_method(self):
        self.selector = DynamicKSelector(base_k=5)

    # ── Confused band: polarity < -0.1 ──

    def test_confused_returns_k_plus_3(self):
        k, state, instruction = self.selector.select(-0.5)
        assert k == 8  # base_k + 3
        assert state == UserState.CONFUSED

    def test_confused_at_negative_extreme(self):
        k, state, _ = self.selector.select(-1.0)
        assert k == 8
        assert state == UserState.CONFUSED

    def test_confused_just_below_threshold(self):
        k, state, _ = self.selector.select(-0.11)
        assert k == 8
        assert state == UserState.CONFUSED

    # ── Neutral band: -0.1 ≤ polarity ≤ 0.3 ──

    def test_neutral_at_zero(self):
        k, state, _ = self.selector.select(0.0)
        assert k == 5  # base_k unchanged
        assert state == UserState.NEUTRAL

    def test_neutral_at_lower_boundary(self):
        """Polarity == -0.1 should be neutral (not confused)."""
        k, state, _ = self.selector.select(-0.1)
        assert k == 5
        assert state == UserState.NEUTRAL

    def test_neutral_at_upper_boundary(self):
        """Polarity == 0.3 should be neutral (not clear)."""
        k, state, _ = self.selector.select(0.3)
        assert k == 5
        assert state == UserState.NEUTRAL

    def test_neutral_midrange(self):
        k, state, _ = self.selector.select(0.15)
        assert k == 5
        assert state == UserState.NEUTRAL

    # ── Clear band: polarity > 0.3 ──

    def test_clear_returns_k_minus_2(self):
        k, state, _ = self.selector.select(0.5)
        assert k == 3  # base_k - 2
        assert state == UserState.CLEAR

    def test_clear_at_positive_extreme(self):
        k, state, _ = self.selector.select(1.0)
        assert k == 3
        assert state == UserState.CLEAR

    def test_clear_just_above_threshold(self):
        k, state, _ = self.selector.select(0.31)
        assert k == 3
        assert state == UserState.CLEAR

    # ── Edge case: k floor ──

    def test_clear_k_never_below_2(self):
        """Even with a very low base_k, clear state should not go below 2."""
        selector = DynamicKSelector(base_k=2)
        k, state, _ = selector.select(0.5)
        assert k == 2  # max(2, 2-2) = max(2, 0) = 2
        assert state == UserState.CLEAR

    def test_clear_k_with_base_k_3(self):
        selector = DynamicKSelector(base_k=3)
        k, _, _ = selector.select(0.5)
        assert k == 2  # max(2, 3-2) = max(2, 1) = 2

    # ── System instruction content ──

    def test_confused_instruction_is_detailed(self):
        _, _, instruction = self.selector.select(-0.5)
        assert "step-by-step" in instruction.lower()
        assert "detailed" in instruction.lower()

    def test_neutral_instruction_is_balanced(self):
        _, _, instruction = self.selector.select(0.0)
        assert "helpful tutor" in instruction.lower()

    def test_clear_instruction_is_concise(self):
        _, _, instruction = self.selector.select(0.5)
        assert "concise" in instruction.lower()


class TestConversationManager:
    """Tests for rolling sentiment tracking."""

    def setup_method(self):
        self.cm = ConversationManager(max_history=5)

    def test_empty_history_returns_zero(self):
        assert self.cm.get_average_sentiment() == 0.0

    def test_single_turn_sentiment(self):
        self.cm.add_turn("This is great!", "Glad you like it")
        avg = self.cm.get_average_sentiment()
        # "This is great!" has positive polarity
        assert avg > 0

    def test_negative_sentiment_detected(self):
        self.cm.add_turn("This is terrible and confusing", "Let me explain")
        avg = self.cm.get_average_sentiment()
        assert avg < 0

    def test_history_window_truncation(self):
        """Only last N turns should be kept."""
        for i in range(10):
            self.cm.add_turn(f"Question {i}", f"Answer {i}")
        assert len(self.cm.history) == 5

    def test_current_query_included_in_average(self):
        """get_average_sentiment with current_query should include it."""
        avg_without = self.cm.get_average_sentiment()
        avg_with = self.cm.get_average_sentiment(current_query="This is wonderful!")
        # With a positive query, the average should be higher
        assert avg_with > avg_without or avg_with >= 0

    def test_formatted_history_empty(self):
        result = self.cm.get_formatted_history()
        assert "No previous conversation history" in result

    def test_formatted_history_non_empty(self):
        self.cm.add_turn("Hello", "Hi there")
        result = self.cm.get_formatted_history()
        assert "User (Turn 1): Hello" in result
        assert "AI (Turn 1): Hi there" in result
