"""
NewsAPI client with retries and client-side rate limiting.

Author: Nnanyelugo Ahukannah

Wraps newsapi.org so the rest of the application never deals with HTTP details.
Everything that can go wrong at the network edge is handled here: rate limits,
transient 5xx, timeouts, and the free tier's habit of returning articles whose
content is truncated or missing entirely.

Run directly to fetch 3 tech headlines -- this is Checkpoint 2:

    python news_api.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from config import config

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

# newsapi.org free tier allows 100 requests/day. One request per second is far
# under any burst limit and keeps us polite; the daily cap is the real ceiling.
MIN_SECONDS_BETWEEN_CALLS = 1.0


class NewsAPIError(RuntimeError):
    """Raised when the news API cannot satisfy a request."""


@dataclass
class Article:
    """A single news article, normalised to the fields this lab actually uses.

    NewsAPI's free tier truncates ``content`` at ~200 characters and sometimes
    omits it, so :meth:`best_text` picks the richest field available rather than
    trusting any single one.
    """

    title: str
    source: str
    url: str
    published_at: str
    description: str | None = None
    content: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "Article":
        """Build an Article from one NewsAPI result, tolerating missing fields."""
        return cls(
            title=(raw.get("title") or "").strip() or "(untitled)",
            source=((raw.get("source") or {}).get("name") or "unknown").strip(),
            url=raw.get("url") or "",
            published_at=raw.get("publishedAt") or "",
            description=(raw.get("description") or None),
            content=(raw.get("content") or None),
        )

    def best_text(self) -> str:
        """Return the richest available text for summarization.

        Prefers ``content`` (longest), falls back to ``description``, then the
        title. Something is always returned so downstream code never has to
        guard against an empty string.
        """
        for candidate in (self.content, self.description, self.title):
            if candidate and candidate.strip():
                return candidate.strip()
        return self.title

    def is_summarizable(self) -> bool:
        """True when there is enough text to be worth spending a token on."""
        return len(self.best_text()) >= 80


class RateLimiter:
    """Minimal client-side spacing between calls.

    Deliberately simple: this enforces a minimum gap rather than a token bucket.
    The free tier's binding constraint is a daily quota, not burst rate, so
    spacing is enough and is easy to reason about in a trace.
    """

    def __init__(self, min_interval: float = MIN_SECONDS_BETWEEN_CALLS) -> None:
        self.min_interval = min_interval
        self._last_call: float | None = None

    def wait(self) -> float:
        """Block until the next call is allowed. Returns seconds actually slept."""
        if self._last_call is None:
            self._last_call = time.monotonic()
            return 0.0
        elapsed = time.monotonic() - self._last_call
        sleep_for = max(0.0, self.min_interval - elapsed)
        if sleep_for:
            time.sleep(sleep_for)
        self._last_call = time.monotonic()
        return sleep_for


class NewsClient:
    """Fetches headlines from newsapi.org."""

    def __init__(self, api_key: str | None = None, timeout: int | None = None) -> None:
        self.api_key = api_key if api_key is not None else config.news_api_key
        self.timeout = timeout if timeout is not None else config.request_timeout
        self.limiter = RateLimiter()
        self.request_count = 0

        if not self.api_key:
            raise NewsAPIError("NEWS_API_KEY is not set - see config.py")

    def fetch_top_headlines(
        self,
        category: str = "technology",
        country: str = "us",
        limit: int = 3,
    ) -> list[Article]:
        """Fetch up to ``limit`` top headlines.

        Retries on transient failures (429 and 5xx) with exponential backoff,
        up to ``config.max_retries``. A 401 is never retried -- a bad key will
        not fix itself, and retrying only burns quota.
        """
        params = {
            "category": category,
            "country": country,
            "pageSize": max(1, min(limit, 100)),
            "apiKey": self.api_key,
        }

        last_error: str = "unknown error"
        for attempt in range(1, config.max_retries + 1):
            self.limiter.wait()
            self.request_count += 1

            try:
                response = requests.get(NEWS_API_URL, params=params, timeout=self.timeout)
            except requests.Timeout:
                last_error = f"timeout after {self.timeout}s"
                self._backoff(attempt)
                continue
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                self._backoff(attempt)
                continue

            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") != "ok":
                    raise NewsAPIError(
                        f"API returned status={payload.get('status')}: "
                        f"{payload.get('message', 'no message')}"
                    )
                articles = [Article.from_api(a) for a in payload.get("articles", [])]
                return articles[:limit]

            if response.status_code == 401:
                # Not retryable: the key is wrong or revoked.
                raise NewsAPIError(
                    "401 Unauthorized - NEWS_API_KEY rejected by newsapi.org. "
                    "Check the key in ~/.config/ironhack/.env.local"
                )

            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                self._backoff(attempt)
                continue

            raise NewsAPIError(f"HTTP {response.status_code}: {response.text[:200]}")

        raise NewsAPIError(
            f"Failed after {config.max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Sleep with exponential backoff: 1s, 2s, 4s..."""
        time.sleep(2 ** (attempt - 1))


# --------------------------------------------------------------------------- #
# Checkpoint 2
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Fetching 3 technology headlines from newsapi.org")
    print("-" * 60)

    client = NewsClient()
    try:
        articles = client.fetch_top_headlines(limit=3)
    except NewsAPIError as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1)

    if not articles:
        print("No articles returned. Try a different category or country.")
        raise SystemExit(1)

    for index, article in enumerate(articles, start=1):
        print(f"\n{index}. {article.title}")
        print(f"   source     : {article.source}")
        print(f"   published  : {article.published_at}")
        print(f"   text chars : {len(article.best_text())}")
        print(f"   summarizable: {'yes' if article.is_summarizable() else 'no (too short)'}")

    print(f"\nOK - {len(articles)} article(s), {client.request_count} API request(s).")
