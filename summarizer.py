"""
Core pipeline: fetch news, summarize on one brain, judge sentiment on the other.

Author: Nnanyelugo Ahukannah

The critical path is deliberately narrow -- one input (a news category), one
transformation (summarize + sentiment), one inspectable output (a ProcessedArticle
record). Each record carries which provider actually did each job, so a reviewer
can see failover when it happens rather than taking it on trust.

Run directly to process 2 articles and print a cost summary -- Checkpoint 4:

    python summarizer.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from llm_providers import CostTracker, ProviderError, ProviderPool
from news_api import Article, NewsAPIError, NewsClient

# Which brain leads which job. Either can cover for the other; these are only
# preferences, and ProviderPool falls back automatically.
SUMMARY_PROVIDER = "cohere"
SENTIMENT_PROVIDER = "openai"


@dataclass
class ProcessedArticle:
    """One article after both brains have seen it.

    ``summary_provider`` and ``sentiment_provider`` record who actually did the
    work, which is the observable evidence that failover occurred.
    """

    title: str
    source: str
    url: str
    published_at: str
    summary: str
    sentiment: str
    summary_provider: str
    sentiment_provider: str
    processed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    error: str | None = None

    def to_dict(self) -> dict:
        """Plain dict, suitable for writing an output record to disk."""
        return asdict(self)

    @property
    def sentiment_label(self) -> str:
        """Just the POSITIVE/NEGATIVE/NEUTRAL label, without the reasoning."""
        first = (self.sentiment or "").strip().splitlines()
        return first[0].strip().upper() if first else "UNKNOWN"


class NewsSummarizer:
    """Ties the news source and the two brains into one workflow."""

    def __init__(
        self,
        news_client: NewsClient | None = None,
        pool: ProviderPool | None = None,
    ) -> None:
        self.tracker = CostTracker()
        self.pool = pool or ProviderPool(self.tracker)
        # Keep the tracker the pool actually uses, in case one was injected.
        self.tracker = self.pool.tracker
        self.news = news_client or NewsClient()

    # ----------------------------------------------------------------- fetch
    def fetch(self, limit: int = 2, category: str = "technology") -> list[Article]:
        """Fetch headlines, keeping only those with enough text to summarize."""
        articles = self.news.fetch_top_headlines(category=category, limit=limit * 2)
        usable = [a for a in articles if a.is_summarizable()]
        return usable[:limit]

    # --------------------------------------------------------------- process
    def process_article(self, article: Article) -> ProcessedArticle:
        """Run both brains over one article.

        A provider failure is captured on the record rather than raised, so one
        bad article cannot abort a whole batch.
        """
        text = article.best_text()
        summary = sentiment = ""
        summary_by = sentiment_by = "none"
        error: str | None = None

        try:
            summary, summary_by = self.pool.run("summarize", text, SUMMARY_PROVIDER)
        except ProviderError as exc:
            error = f"summarize failed: {exc}"
            summary = "(unavailable)"

        try:
            # Sentiment reads the summary when we have one -- it is cleaner input
            # than a truncated blurb, and it chains the two brains together,
            # which is the point of the exercise.
            sentiment_input = summary if summary and summary != "(unavailable)" else text
            sentiment, sentiment_by = self.pool.run(
                "sentiment", sentiment_input, SENTIMENT_PROVIDER
            )
        except ProviderError as exc:
            error = (error + " | " if error else "") + f"sentiment failed: {exc}"
            sentiment = "(unavailable)"

        return ProcessedArticle(
            title=article.title,
            source=article.source,
            url=article.url,
            published_at=article.published_at,
            summary=summary,
            sentiment=sentiment,
            summary_provider=summary_by,
            sentiment_provider=sentiment_by,
            error=error,
        )

    def process(self, limit: int = 2, category: str = "technology") -> list[ProcessedArticle]:
        """Fetch and process ``limit`` articles sequentially."""
        return [self.process_article(a) for a in self.fetch(limit, category)]

    # ----------------------------------------------------------------- async
    async def process_async(
        self, limit: int = 2, category: str = "technology"
    ) -> list[ProcessedArticle]:
        """Concurrent variant of :meth:`process` (lab Part 5, optional).

        The provider SDKs used here are synchronous, so each article is run in a
        worker thread rather than with native async I/O. The win is real -- calls
        overlap instead of queueing -- without pretending the clients are async.
        """
        articles = self.fetch(limit, category)
        tasks = [asyncio.to_thread(self.process_article, a) for a in articles]
        return list(await asyncio.gather(*tasks))

    # ---------------------------------------------------------------- report
    def cost_summary(self) -> str:
        return self.tracker.summary()


def render(records: list[ProcessedArticle]) -> str:
    """Format processed articles for a terminal reader."""
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        lines.append(f"\n{index}. {record.title}")
        lines.append(f"   source    : {record.source}")
        lines.append(f"   summary   : ({record.summary_provider}) {record.summary}")
        lines.append(
            f"   sentiment : ({record.sentiment_provider}) {record.sentiment_label}"
        )
        if record.error:
            lines.append(f"   ERROR     : {record.error}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Checkpoint 4
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Processing 2 technology articles")
    print("=" * 68)

    try:
        summarizer = NewsSummarizer()
        processed = summarizer.process(limit=2)
    except (NewsAPIError, ProviderError) as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1)

    if not processed:
        print("No summarizable articles returned.")
        raise SystemExit(1)

    print(render(processed))
    print()
    print(summarizer.cost_summary())

    if summarizer.pool.failovers:
        print("\nFailover events:")
        for note in summarizer.pool.failovers:
            print(f"  - {note}")
    else:
        print("\nNo failover needed - both providers healthy.")

    print(f"\nOK - {len(processed)} article(s) processed by two providers.")
