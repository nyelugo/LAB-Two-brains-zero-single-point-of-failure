"""
Configuration and secret loading for the multi-provider news summarizer.

Author: Nnanyelugo Ahukannah

Secrets are read from the shared Ironhack env file rather than a per-lab .env,
so one file serves every lab and no key is ever copied into a repo:

    ~/.config/ironhack/.env.local      (mode 600, outside every git repo)

A per-lab .env is still honoured if present, and overrides the shared file. That
keeps the door open for lab-specific values without duplicating the real keys.

Run directly to validate the environment -- this is Checkpoint 1:

    python config.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Environment loading
# --------------------------------------------------------------------------- #

SHARED_ENV = Path.home() / ".config" / "ironhack" / ".env.local"
LOCAL_ENV = Path(__file__).parent / ".env"


def load_environment() -> None:
    """Load the shared Ironhack env file, then any per-lab overrides.

    Order matters: the shared file is loaded first, then the local .env with
    ``override=True`` so a lab-specific value wins if one is defined.
    """
    if SHARED_ENV.exists():
        load_dotenv(SHARED_ENV)
    if LOCAL_ENV.exists():
        load_dotenv(LOCAL_ENV, override=True)


load_environment()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def _get_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back on anything unparseable."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Runtime configuration, assembled from the environment.

    Two LLM providers are configured deliberately -- that is the whole point of
    this lab. One summarizes, the other analyses sentiment, and either can stand
    in for the other when a provider fails.
    """

    # Secrets
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))

    # Operational settings
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    max_retries: int = field(default_factory=lambda: _get_int("MAX_RETRIES", 3))
    request_timeout: int = field(default_factory=lambda: _get_int("REQUEST_TIMEOUT", 30))
    daily_budget: float = field(default_factory=lambda: _get_float("DAILY_BUDGET", 5.00))

    # Required for the lab to function at all.
    REQUIRED = ("COHERE_API_KEY", "OPENAI_API_KEY", "NEWS_API_KEY")

    def missing_keys(self) -> list[str]:
        """Return the names of required keys that are absent or empty."""
        present = {
            "COHERE_API_KEY": self.cohere_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "NEWS_API_KEY": self.news_api_key,
        }
        return [name for name, value in present.items() if not value.strip()]

    def validate(self) -> None:
        """Raise :class:`ConfigError` naming every missing key at once.

        Reporting all of them together beats failing on the first one -- the
        user fixes the environment in a single pass.
        """
        missing = self.missing_keys()
        if missing:
            raise ConfigError(
                "Missing required configuration: "
                + ", ".join(missing)
                + f"\nAdd them to {SHARED_ENV} (mode 600)."
            )

    @staticmethod
    def fingerprint(value: str) -> str:
        """Return a short, non-reversible fingerprint of a secret.

        Used so startup output can prove a key is loaded without ever printing
        the key itself.
        """
        import hashlib

        if not value:
            return "unset"
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        return f"len={len(value)} fp={digest}"


config = Config()


# --------------------------------------------------------------------------- #
# Checkpoint 1
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Configuration check")
    print("-" * 48)
    print(f"shared env : {SHARED_ENV}  {'found' if SHARED_ENV.exists() else 'MISSING'}")
    print(f"local env  : {LOCAL_ENV}  {'found' if LOCAL_ENV.exists() else 'not present (fine)'}")
    print()

    for key_name, secret in (
        ("COHERE_API_KEY", config.cohere_api_key),
        ("OPENAI_API_KEY", config.openai_api_key),
        ("NEWS_API_KEY", config.news_api_key),
    ):
        print(f"  {key_name:<16} {Config.fingerprint(secret)}")

    print()
    print(f"  environment      {config.environment}")
    print(f"  max_retries      {config.max_retries}")
    print(f"  request_timeout  {config.request_timeout}s")
    print(f"  daily_budget     ${config.daily_budget:.2f}")
    print()

    try:
        config.validate()
    except ConfigError as exc:
        print(f"FAILED\n{exc}")
        raise SystemExit(1)

    print("OK - all required configuration present.")
