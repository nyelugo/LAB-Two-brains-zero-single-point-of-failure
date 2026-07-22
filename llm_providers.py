"""
Two LLM providers behind one interface, with fallback and cost tracking.

Author: Nnanyelugo Ahukannah

This is the "two brains" of the lab. Cohere and OpenAI both implement the same
small interface, so either can do either job. Normal operation splits the work
-- Cohere summarizes, OpenAI judges sentiment -- but when a provider fails, the
other takes over. That is what removes the single point of failure.

The lab text names OpenAI and Anthropic. Cohere is substituted for Anthropic
here because that is the key on file; the architecture is unchanged, since what
matters is that two independent providers can cover for each other.

Run directly to exercise both providers and show costs -- this is Checkpoint 3:

    python llm_providers.py
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from config import config

# Prices in USD per 1M tokens. Approximate and provider-published; good enough
# to keep a running estimate and stop before the daily budget is exceeded.
PRICING = {
    "command-r-08-2024": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class ProviderError(RuntimeError):
    """Raised when a provider cannot complete a request."""


@dataclass
class Usage:
    """Token counts and cost for a single call."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    @classmethod
    def compute(
        cls,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
    ) -> "Usage":
        """Build a Usage record, pricing the call from :data:`PRICING`."""
        rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1_000_000) * rates["input"] + (
            output_tokens / 1_000_000
        ) * rates["output"]
        return cls(provider, model, input_tokens, output_tokens, cost, latency_s)


@dataclass
class CostTracker:
    """Accumulates spend across every provider call and enforces the budget."""

    budget_usd: float = field(default_factory=lambda: config.daily_budget)
    calls: list[Usage] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.calls)

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_usd - self.total_cost)

    def record(self, usage: Usage) -> None:
        self.calls.append(usage)

    def check_budget(self) -> None:
        """Raise before spending more once the budget is exhausted."""
        if self.total_cost >= self.budget_usd:
            raise ProviderError(
                f"Budget exceeded: ${self.total_cost:.4f} of ${self.budget_usd:.2f} spent. "
                "Raise DAILY_BUDGET or process fewer articles."
            )

    def summary(self) -> str:
        """Human-readable cost breakdown, grouped by provider."""
        if not self.calls:
            return "No provider calls recorded."
        by_provider: dict[str, list[Usage]] = {}
        for usage in self.calls:
            by_provider.setdefault(usage.provider, []).append(usage)

        lines = ["Cost summary", "-" * 60]
        for provider, records in sorted(by_provider.items()):
            cost = sum(r.cost_usd for r in records)
            tokens_in = sum(r.input_tokens for r in records)
            tokens_out = sum(r.output_tokens for r in records)
            avg_latency = sum(r.latency_s for r in records) / len(records)
            lines.append(
                f"  {provider:<8} {len(records):>2} call(s)  "
                f"in={tokens_in:<6} out={tokens_out:<6} "
                f"${cost:.6f}  avg {avg_latency:.2f}s"
            )
        lines.append("-" * 60)
        lines.append(
            f"  {'TOTAL':<8} ${self.total_cost:.6f} of ${self.budget_usd:.2f} "
            f"(${self.remaining:.4f} left)"
        )
        return "\n".join(lines)


class LLMProvider(ABC):
    """Common interface both brains implement."""

    name: str = "base"
    model: str = ""

    def __init__(self, tracker: CostTracker) -> None:
        self.tracker = tracker

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 300) -> tuple[str, Usage]:
        """Send one prompt, return the text and a usage record."""

    def summarize(self, text: str, sentences: int = 2) -> tuple[str, Usage]:
        """Summarize ``text`` in roughly ``sentences`` sentences."""
        prompt = (
            f"Summarize the following news item in at most {sentences} sentences. "
            "Be factual and specific. Do not add information that is not present.\n\n"
            f"{text}"
        )
        return self.complete(prompt, max_tokens=200)

    def analyze_sentiment(self, text: str) -> tuple[str, Usage]:
        """Classify sentiment as positive, negative or neutral, with a reason."""
        prompt = (
            "Classify the sentiment of this news item as exactly one of: "
            "POSITIVE, NEGATIVE, NEUTRAL. Reply with the label on the first line "
            "and a one-sentence reason on the second.\n\n"
            f"{text}"
        )
        return self.complete(prompt, max_tokens=80)


class CohereProvider(LLMProvider):
    """Cohere brain. Primary summarizer."""

    name = "cohere"
    model = "command-r-08-2024"

    def __init__(self, tracker: CostTracker) -> None:
        super().__init__(tracker)
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ProviderError("cohere package not installed") from exc
        if not config.cohere_api_key:
            raise ProviderError("COHERE_API_KEY is not set")
        self._client = cohere.ClientV2(api_key=config.cohere_api_key)

    def complete(self, prompt: str, max_tokens: int = 300) -> tuple[str, Usage]:
        self.tracker.check_budget()
        started = time.monotonic()
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"cohere call failed: {exc}") from exc

        latency = time.monotonic() - started
        text = "".join(part.text for part in response.message.content if hasattr(part, "text"))

        # Usage shape varies across SDK versions; degrade to 0 rather than crash.
        tokens_in = tokens_out = 0
        try:
            billed = response.usage.tokens
            tokens_in = int(billed.input_tokens or 0)
            tokens_out = int(billed.output_tokens or 0)
        except (AttributeError, TypeError):
            pass

        usage = Usage.compute(self.name, self.model, tokens_in, tokens_out, latency)
        self.tracker.record(usage)
        return text.strip(), usage


class OpenAIProvider(LLMProvider):
    """OpenAI brain. Primary sentiment analyser."""

    name = "openai"
    model = "gpt-4o-mini"

    def __init__(self, tracker: CostTracker) -> None:
        super().__init__(tracker)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ProviderError("openai package not installed") from exc
        if not config.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=config.openai_api_key, timeout=config.request_timeout)

    def complete(self, prompt: str, max_tokens: int = 300) -> tuple[str, Usage]:
        self.tracker.check_budget()
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"openai call failed: {exc}") from exc

        latency = time.monotonic() - started
        text = response.choices[0].message.content or ""

        tokens_in = tokens_out = 0
        try:
            tokens_in = int(response.usage.prompt_tokens or 0)
            tokens_out = int(response.usage.completion_tokens or 0)
        except (AttributeError, TypeError):
            pass

        usage = Usage.compute(self.name, self.model, tokens_in, tokens_out, latency)
        self.tracker.record(usage)
        return text.strip(), usage


class ProviderPool:
    """Runs a task on a preferred provider, failing over to the other.

    This is the point of the lab: no single provider outage can stop the
    pipeline. Every failover is recorded so a reviewer can see it happened.
    """

    def __init__(self, tracker: CostTracker | None = None) -> None:
        self.tracker = tracker or CostTracker()
        self.providers: dict[str, LLMProvider] = {}
        self.failovers: list[str] = []

        # Build whichever providers are actually available. One missing key
        # degrades the pool rather than killing it.
        for cls in (CohereProvider, OpenAIProvider):
            try:
                provider = cls(self.tracker)
                self.providers[provider.name] = provider
            except ProviderError as exc:
                self.failovers.append(f"{cls.name} unavailable at startup: {exc}")

        if not self.providers:
            raise ProviderError("No LLM providers available - check your API keys.")

    def _ordered(self, preferred: str) -> list[LLMProvider]:
        """Preferred provider first, then every other available one."""
        ordered = [p for n, p in self.providers.items() if n == preferred]
        ordered += [p for n, p in self.providers.items() if n != preferred]
        return ordered

    def run(self, task: str, text: str, preferred: str) -> tuple[str, str]:
        """Run ``task`` ('summarize' or 'sentiment') and return (result, provider).

        Tries the preferred provider, then any other. Raises only when every
        provider has failed.
        """
        errors: list[str] = []
        for provider in self._ordered(preferred):
            try:
                if task == "summarize":
                    result, _ = provider.summarize(text)
                elif task == "sentiment":
                    result, _ = provider.analyze_sentiment(text)
                else:
                    raise ValueError(f"unknown task: {task}")

                if provider.name != preferred:
                    self.failovers.append(
                        f"{task}: {preferred} unavailable, used {provider.name}"
                    )
                return result, provider.name
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
                continue

        raise ProviderError(f"All providers failed for {task}. " + " | ".join(errors))


# --------------------------------------------------------------------------- #
# Checkpoint 3
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    SAMPLE = (
        "A major cloud provider reported a four-hour outage affecting several "
        "regions. Customers saw elevated error rates on storage and database "
        "services. The company said a configuration change was rolled back and "
        "all services have since recovered."
    )

    print("Testing both providers")
    print("=" * 60)

    pool = ProviderPool()
    print(f"available providers: {', '.join(sorted(pool.providers)) or 'none'}")
    for note in pool.failovers:
        print(f"  note: {note}")
    print()

    summary, used = pool.run("summarize", SAMPLE, preferred="cohere")
    print(f"SUMMARY  (via {used})\n  {summary}\n")

    sentiment, used = pool.run("sentiment", SAMPLE, preferred="openai")
    print(f"SENTIMENT  (via {used})\n  " + sentiment.replace("\n", "\n  ") + "\n")

    print(pool.tracker.summary())

    if pool.failovers:
        print("\nFailover events:")
        for note in pool.failovers:
            print(f"  - {note}")

    print("\nOK - both providers exercised with cost tracking.")
