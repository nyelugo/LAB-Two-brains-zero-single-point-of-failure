# Proof of Learning

**LAB | Two brains, zero single point of failure**
Author: Nnanyelugo Ahukannah

Reviewer entry point. Everything below was produced by a real run.

## 1. Workflow configuration

```
INPUT    news category (default: technology)
DECIDE   fetch top headlines; keep only articles with >= 80 chars of text
BUILD    NewsAPI -> Cohere (summarize) -> OpenAI (sentiment) -> JSON records
VERIFY   artifacts/run_trace.json records every LLM call, cost, and failover event
EXPLAIN  see "Operational risk" below
```

Provider roles are preferences, not hard wiring — `ProviderPool` falls back
automatically. Configuration lives in `config.py`; secrets are read from
`~/.config/ironhack/.env.local`, outside the repo.

## 2. Execution trace

- `artifacts/run_trace.json` — healthy run, 4 articles, 8 LLM calls, 0 failovers
- `artifacts/run_trace_failover.json` — Cohere forcibly offline, 3 failover events recorded

Reproduce either:

```bash
python main.py --auto -n 4
python main.py --auto -n 2 --simulate-outage cohere
```

## 3. Input payload

One article as received from NewsAPI (`content` truncated by the free tier):

```json
{
  "source": { "name": "Motor1" },
  "title": "The New Mercedes-Maybach GLS Debuts With More Power, Better Style",
  "publishedAt": "2026-07-21T03:00:23Z",
  "content": "Mercedes-Benz updated the GLS earlier this year, and now the Maybach variant... [+214 chars]"
}
```

## 4. Output record

```json
{
  "title": "The New Mercedes-Maybach GLS Debuts With More Power, Better Style",
  "source": "Motor1",
  "summary": "Mercedes-Benz is set to launch the 2027 Maybach GLS 680, an updated three-row SUV, later this year.",
  "sentiment": "POSITIVE",
  "summary_provider": "cohere",
  "sentiment_provider": "openai"
}
```

Full records in `artifacts/output_records.json`.

## 5. Verification

| checkpoint | command | result |
|---|---|---|
| 1 | `python config.py` | all three keys load |
| 2 | `python news_api.py` | 3 live headlines |
| 3 | `python llm_providers.py` | both providers respond, costs tracked |
| 4 | `python summarizer.py` | 2 articles summarized + classified |
| 5 | `pytest test_summarizer.py -v` | **29 passed** |
| 6 | `python main.py --auto -n 4` | 4 articles, artifacts written |

Screenshots of the verifying runs:

- `screenshots/pytest_passing.png` — 29 passed
- `screenshots/main_app_running.png` — 4 articles processed
- `screenshots/failover_demo.png` — Cohere offline, every job served by OpenAI

`summary_provider` and `sentiment_provider` on every output record are the
observable evidence of which brain did which job.

## 6. Operational risk

**First failure mode I would monitor if this ran daily: silent quality collapse
under sustained failover.** A provider outage is loud and already handled. The
dangerous case is one provider degrading — rate-limited, slow, or returning
truncated answers — so every job lands on the survivor. Throughput still looks
fine and no error is raised, but all output now comes from one model, cost on
that provider roughly doubles, and the redundancy the system claims no longer
exists. I would alert on failover *rate* rather than failover *events*: a
sustained non-zero rate means the system is running without a spare.

Secondary risks: the NewsAPI free tier's 100 requests/day cap, which would fail
mid-run with no warning; and cost drift, since figures here are computed from
published rates, not billed amounts.
