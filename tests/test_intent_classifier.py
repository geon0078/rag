"""Intent classifier contract tests.

Covers `_normalize` (verdict parsing) and a mocked `classify_sync` path. The
OpenAI client is stubbed so the test is hermetic; the live model behavior is
covered by `scripts/eval_supplementary.py` (`negative_rejection`).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.generation.intent_classifier import IntentClassifier, _normalize


class TestNormalize:
    def test_exact_answerable(self) -> None:
        assert _normalize("answerable") == "answerable"

    def test_exact_unanswerable(self) -> None:
        assert _normalize("unanswerable") == "unanswerable"

    def test_strips_whitespace_and_case(self) -> None:
        assert _normalize("  Answerable\n") == "answerable"
        assert _normalize("UNANSWERABLE") == "unanswerable"

    def test_substring_unanswerable_wins_when_present(self) -> None:
        assert _normalize("verdict: unanswerable, reason=...") == "unanswerable"

    def test_substring_answerable(self) -> None:
        assert _normalize("This is answerable I think") == "answerable"

    def test_unrecognized_defaults_to_answerable(self) -> None:
        assert _normalize("maybe") == "answerable"
        assert _normalize("") == "answerable"


def _make_classifier_with_response(verdict: str) -> IntentClassifier:
    classifier = IntentClassifier()
    mock_choice = MagicMock()
    mock_choice.message.content = verdict
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    classifier.sync_client = MagicMock()
    classifier.sync_client.chat.completions.create.return_value = mock_response
    return classifier


@pytest.mark.unit
class TestClassifySyncMocked:
    """4 negative + 4 positive cases through the mocked sync path."""

    @pytest.mark.parametrize(
        "query",
        [
            "내 비밀번호 뭐였지?",
            "버스 시간표 알려줘",
            "...",
            "그거 어떻게 해?",
        ],
    )
    def test_unanswerable_queries(self, query: str) -> None:
        classifier = _make_classifier_with_response("unanswerable")
        assert classifier.classify_sync(query) == "unanswerable"

    @pytest.mark.parametrize(
        "query",
        [
            "수강신청 언제 해?",
            "간호학과 졸업 학점은?",
            "성남캠퍼스에 OO학과 있어?",
            "장학금 신청 절차 알려줘",
        ],
    )
    def test_answerable_queries(self, query: str) -> None:
        classifier = _make_classifier_with_response("answerable")
        assert classifier.classify_sync(query) == "answerable"

    def test_latency_budget_constraints(self) -> None:
        """Verdict must fit in <= 5 tokens with deterministic decoding."""
        classifier = _make_classifier_with_response("answerable")
        classifier.classify_sync("test")
        kwargs = classifier.sync_client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] <= 5
        assert kwargs["temperature"] == 0.0
