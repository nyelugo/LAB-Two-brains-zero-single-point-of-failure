"""
Unit tests for the multi-provider news summarizer.

Author: Nnanyelugo Ahukannah

Every test here runs offline. Providers and HTTP are mocked, so the suite is
fast, free, and gives the same answer whether or not the network is up -- a
test that needs a live API key is not a unit test.

The tests that matter most are the failover ones: they are the only evidence
that "zero single point of failure" is true rather than merely intended.

    pytest test_summarizer.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import requests

from llm_providers import CostTracker, LLMProvider, ProviderError, ProviderPool, Usage
from news_api import Article, NewsAPIError, NewsClient, RateLimiter
from summarizer import NewsSummarizer, ProcessedArticle


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeProvider(LLMProvider):
    """A provider that answers instantly, or fails on demand."""

    def __init__(self, tracker, name="fake", should_fail=False):
        self.name = name
        self.model = "fake-model"
        self.should_fail = should_fail
        self.calls = 0
        super().__init__(tracker)

    def complete(self, prompt: str, max_tokens: int = 300):
        self.calls += 1
        if self.should_fail:
            raise ProviderError(f"{self.name} is down")
        usage = Usage.compute(self.name, self.model, 10, 5, 0.01)
        self.tracker.record(usage)
        return f"{self.name} handled it", usage


def make_pool(cohere_fails=False, openai_fails=False) -> ProviderPool:
    """Build a ProviderPool with fakes, bypassing real SDK construction."""
    pool = ProviderPool.__new__(ProviderPool)
    pool.tracker = CostTracker()
    pool.failovers = []
    pool.providers = {
        "cohere": FakeProvider(pool.tracker, "cohere", cohere_fails),
        "openai": FakeProvider(pool.tracker, "openai", openai_fails),
    }
    return pool


# --------------------------------------------------------------------------- #
# Article
# --------------------------------------------------------------------------- #


class TestArticle:
    def test_best_text_prefers_content(self):
        article = Article("T", "S", "u", "d", description="desc", content="content here")
        assert article.best_text() == "content here"

    def test_best_text_falls_back_to_description(self):
        article = Article("T", "S", "u", "d", description="desc", content=None)
        assert article.best_text() == "desc"

    def test_best_text_falls_back_to_title(self):
        article = Article("Only title", "S", "u", "d")
        assert article.best_text() == "Only title"

    def test_best_text_ignores_whitespace_only_fields(self):
        """A field of spaces is not content -- this bit NewsAPI's free tier."""
        article = Article("Title", "S", "u", "d", description="   ", content="")
        assert article.best_text() == "Title"

    def test_is_summarizable_rejects_short_text(self):
        assert not Article("tiny", "S", "u", "d").is_summarizable()

    def test_is_summarizable_accepts_long_text(self):
        assert Article("T", "S", "u", "d", content="x" * 100).is_summarizable()

    def test_from_api_tolerates_missing_fields(self):
        article = Article.from_api({})
        assert article.title == "(untitled)"
        assert article.source == "unknown"


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #


class TestRateLimiter:
    def test_first_call_does_not_wait(self):
        assert RateLimiter(min_interval=5.0).wait() == 0.0

    def test_second_call_waits(self):
        limiter = RateLimiter(min_interval=0.05)
        limiter.wait()
        assert limiter.wait() > 0


# --------------------------------------------------------------------------- #
# Cost tracking
# --------------------------------------------------------------------------- #


class TestCostTracking:
    def test_cost_is_priced_per_million_tokens(self):
        usage = Usage.compute("openai", "gpt-4o-mini", 1_000_000, 0, 0.1)
        assert usage.cost_usd == pytest.approx(0.15)

    def test_unknown_model_costs_zero_rather_than_crashing(self):
        assert Usage.compute("x", "no-such-model", 1000, 1000, 0.1).cost_usd == 0.0

    def test_tracker_accumulates(self):
        tracker = CostTracker(budget_usd=1.0)
        tracker.record(Usage.compute("a", "gpt-4o-mini", 1_000_000, 0, 0.1))
        tracker.record(Usage.compute("b", "gpt-4o-mini", 1_000_000, 0, 0.1))
        assert tracker.total_cost == pytest.approx(0.30)
        assert tracker.remaining == pytest.approx(0.70)

    def test_budget_check_raises_when_exhausted(self):
        tracker = CostTracker(budget_usd=0.10)
        tracker.record(Usage.compute("a", "gpt-4o-mini", 1_000_000, 0, 0.1))
        with pytest.raises(ProviderError, match="Budget exceeded"):
            tracker.check_budget()


# --------------------------------------------------------------------------- #
# Failover -- the core claim of this lab
# --------------------------------------------------------------------------- #


class TestFailover:
    def test_preferred_provider_used_when_healthy(self):
        pool = make_pool()
        result, used = pool.run("summarize", "some text", preferred="cohere")
        assert used == "cohere"
        assert pool.failovers == []

    def test_falls_over_when_preferred_provider_fails(self):
        pool = make_pool(cohere_fails=True)
        result, used = pool.run("summarize", "some text", preferred="cohere")
        assert used == "openai"
        assert "cohere unavailable" in pool.failovers[0]

    def test_failover_is_recorded_for_the_reviewer(self):
        """Failover must leave a trace, not happen silently."""
        pool = make_pool(openai_fails=True)
        pool.run("sentiment", "text", preferred="openai")
        assert len(pool.failovers) == 1
        assert "used cohere" in pool.failovers[0]

    def test_raises_only_when_every_provider_fails(self):
        pool = make_pool(cohere_fails=True, openai_fails=True)
        with pytest.raises(ProviderError, match="All providers failed"):
            pool.run("summarize", "text", preferred="cohere")

    def test_unknown_task_is_rejected(self):
        pool = make_pool()
        with pytest.raises((ValueError, ProviderError)):
            pool.run("translate", "text", preferred="cohere")


# --------------------------------------------------------------------------- #
# NewsClient HTTP behaviour
# --------------------------------------------------------------------------- #


class TestNewsClient:
    def _response(self, status=200, payload=None):
        response = MagicMock()
        response.status_code = status
        response.json.return_value = payload or {"status": "ok", "articles": []}
        response.text = "body"
        return response

    def test_missing_key_raises(self):
        with pytest.raises(NewsAPIError, match="NEWS_API_KEY"):
            NewsClient(api_key="")

    @patch("news_api.requests.get")
    def test_successful_fetch_returns_articles(self, mock_get):
        mock_get.return_value = self._response(
            payload={
                "status": "ok",
                "articles": [
                    {"title": "A", "source": {"name": "S"}, "url": "u", "content": "x" * 90}
                ],
            }
        )
        articles = NewsClient(api_key="k").fetch_top_headlines(limit=1)
        assert len(articles) == 1 and articles[0].title == "A"

    @patch("news_api.requests.get")
    def test_401_is_not_retried(self, mock_get):
        """A bad key will not fix itself; retrying only burns quota."""
        mock_get.return_value = self._response(status=401)
        with pytest.raises(NewsAPIError, match="401"):
            NewsClient(api_key="bad").fetch_top_headlines()
        assert mock_get.call_count == 1

    @patch("news_api.NewsClient._backoff", lambda *_: None)
    @patch("news_api.requests.get")
    def test_500_is_retried_then_gives_up(self, mock_get):
        mock_get.return_value = self._response(status=500)
        with pytest.raises(NewsAPIError, match="Failed after"):
            NewsClient(api_key="k").fetch_top_headlines()
        assert mock_get.call_count > 1

    @patch("news_api.NewsClient._backoff", lambda *_: None)
    @patch("news_api.requests.get", side_effect=requests.Timeout())
    def test_timeout_is_retried(self, mock_get):
        with pytest.raises(NewsAPIError):
            NewsClient(api_key="k").fetch_top_headlines()
        assert mock_get.call_count > 1


# --------------------------------------------------------------------------- #
# Summarizer pipeline
# --------------------------------------------------------------------------- #


class TestNewsSummarizer:
    def _summarizer(self, pool=None):
        summarizer = NewsSummarizer.__new__(NewsSummarizer)
        summarizer.pool = pool or make_pool()
        summarizer.tracker = summarizer.pool.tracker
        summarizer.news = MagicMock()
        return summarizer

    def test_process_article_records_both_providers(self):
        summarizer = self._summarizer()
        record = summarizer.process_article(Article("T", "S", "u", "d", content="x" * 100))
        assert record.summary_provider == "cohere"
        assert record.sentiment_provider == "openai"
        assert record.error is None

    def test_one_dead_provider_does_not_abort_the_article(self):
        summarizer = self._summarizer(make_pool(cohere_fails=True))
        record = summarizer.process_article(Article("T", "S", "u", "d", content="x" * 100))
        assert record.summary_provider == "openai"
        assert record.error is None

    def test_total_outage_is_captured_on_the_record_not_raised(self):
        summarizer = self._summarizer(make_pool(cohere_fails=True, openai_fails=True))
        record = summarizer.process_article(Article("T", "S", "u", "d", content="x" * 100))
        assert record.summary == "(unavailable)"
        assert "summarize failed" in record.error

    def test_fetch_filters_out_unsummarizable_articles(self):
        summarizer = self._summarizer()
        summarizer.news.fetch_top_headlines.return_value = [
            Article("short", "S", "u", "d"),
            Article("long", "S", "u", "d", content="x" * 100),
        ]
        assert [a.title for a in summarizer.fetch(limit=2)] == ["long"]

    def test_sentiment_label_extracts_first_line(self):
        record = ProcessedArticle(
            "t", "s", "u", "d", "sum", "POSITIVE\nbecause reasons", "c", "o"
        )
        assert record.sentiment_label == "POSITIVE"

    def test_to_dict_is_serialisable(self):
        record = ProcessedArticle("t", "s", "u", "d", "sum", "NEUTRAL", "c", "o")
        assert record.to_dict()["summary_provider"] == "c"
