"""
Interactive entry point for the multi-provider news summarizer.

Author: Nnanyelugo Ahukannah

Runs the workflow end to end and writes an execution trace a reviewer can
inspect. Interactive by default; ``--auto`` runs the same pipeline without
prompting, which is what makes the run reproducible for screenshots and CI.

    python main.py                 # interactive
    python main.py --auto -n 4     # non-interactive, 4 articles
    python main.py --auto --async  # concurrent processing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_providers import ProviderError
from news_api import NewsAPIError
from summarizer import NewsSummarizer, render

CATEGORIES = ["technology", "business", "science", "health", "sports", "general"]
TRACE_PATH = Path(__file__).parent / "run_trace.json"
OUTPUT_PATH = Path(__file__).parent / "output_records.json"

BANNER = r"""
+--------------------------------------------------------------+
|  Multi-Provider News Summarizer                              |
|  Cohere summarizes  ->  OpenAI judges sentiment              |
|  Either brain covers for the other. No single point of failure|
+--------------------------------------------------------------+
"""


def choose_category() -> str:
    """Prompt for a news category, defaulting to technology."""
    print("\nCategories:")
    for index, name in enumerate(CATEGORIES, start=1):
        print(f"  {index}. {name}")
    raw = input(f"\nChoose 1-{len(CATEGORIES)} [1]: ").strip()
    if not raw:
        return CATEGORIES[0]
    try:
        return CATEGORIES[int(raw) - 1]
    except (ValueError, IndexError):
        print("Not a valid choice - using technology.")
        return CATEGORIES[0]


def choose_count() -> int:
    """Prompt for how many articles to process, 3-5 per the lab."""
    raw = input("How many articles? 3-5 [3]: ").strip()
    if not raw:
        return 3
    try:
        return max(1, min(int(raw), 10))
    except ValueError:
        print("Not a number - using 3.")
        return 3


def write_artifacts(summarizer: NewsSummarizer, records, category: str, concurrent: bool) -> None:
    """Write the execution trace and output records the lab asks for."""
    trace = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": category,
        "mode": "async" if concurrent else "sync",
        "articles_requested": len(records),
        "articles_processed": len(records),
        "providers_available": sorted(summarizer.pool.providers),
        "failover_events": summarizer.pool.failovers,
        "news_api_requests": summarizer.news.request_count,
        "llm_calls": [
            {
                "provider": u.provider,
                "model": u.model,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cost_usd": round(u.cost_usd, 8),
                "latency_s": round(u.latency_s, 3),
            }
            for u in summarizer.tracker.calls
        ],
        "total_cost_usd": round(summarizer.tracker.total_cost, 8),
        "budget_usd": summarizer.tracker.budget_usd,
    }
    TRACE_PATH.write_text(json.dumps(trace, indent=2))
    OUTPUT_PATH.write_text(json.dumps([r.to_dict() for r in records], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-provider news summarizer")
    parser.add_argument("--auto", action="store_true", help="run without prompting")
    parser.add_argument("-n", "--count", type=int, default=3, help="articles to process")
    parser.add_argument("-c", "--category", default="technology", choices=CATEGORIES)
    parser.add_argument(
        "--async", dest="concurrent", action="store_true", help="process concurrently"
    )
    parser.add_argument(
        "--simulate-outage",
        choices=["cohere", "openai"],
        help="disable a provider to demonstrate failover on a real run",
    )
    args = parser.parse_args()

    print(BANNER)

    if args.auto:
        category, count = args.category, args.count
        print(f"Auto mode: {count} article(s) from '{category}'"
              f"{' (concurrent)' if args.concurrent else ''}")
    else:
        category = choose_category()
        count = choose_count()

    try:
        summarizer = NewsSummarizer()
    except ProviderError as exc:
        print(f"\nStartup failed: {exc}")
        return 1

    if args.simulate_outage:
        # Pull a provider out of the pool at runtime. This is how the fallback
        # path gets exercised against the real API rather than only in tests.
        removed = summarizer.pool.providers.pop(args.simulate_outage, None)
        if removed is not None:
            summarizer.pool.failovers.append(
                f"simulated outage: {args.simulate_outage} removed from pool"
            )
            print(f"\n*** SIMULATED OUTAGE: {args.simulate_outage} is offline ***")

    print(f"\nProviders online: {', '.join(sorted(summarizer.pool.providers))}")
    print(f"Fetching {count} article(s) from '{category}'...\n")

    try:
        if args.concurrent:
            records = asyncio.run(summarizer.process_async(limit=count, category=category))
        else:
            records = summarizer.process(limit=count, category=category)
    except NewsAPIError as exc:
        print(f"News fetch failed: {exc}")
        return 1
    except ProviderError as exc:
        print(f"All providers failed: {exc}")
        return 1

    if not records:
        print("No summarizable articles were returned. Try another category.")
        return 1

    print("=" * 68)
    print(render(records))
    print()
    print("=" * 68)
    print(summarizer.cost_summary())

    if summarizer.pool.failovers:
        print("\nFailover events (proof the fallback path works):")
        for note in summarizer.pool.failovers:
            print(f"  - {note}")
    else:
        print("\nNo failover needed - both providers healthy this run.")

    write_artifacts(summarizer, records, category, args.concurrent)
    print(f"\nWrote {TRACE_PATH.name} and {OUTPUT_PATH.name}")
    print(f"OK - {len(records)} article(s) processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
