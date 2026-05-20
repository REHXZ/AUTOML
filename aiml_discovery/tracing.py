"""Phoenix / OpenTelemetry tracing for the aiml_discovery agent system.

Call configure_tracing() once at app startup. Fails gracefully — if
arize-phoenix-otel or openinference-instrumentation-openai are not installed
the app runs without tracing and logs a warning instead of crashing.

Env vars (all optional):
  PHOENIX_PROJECT_NAME         default: AIML-Discovery
  PHOENIX_COLLECTOR_ENDPOINT   default: http://localhost:4317
  PHOENIX_TRACING_ENABLED      set to "false" to skip tracing entirely
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_configured = False


def configure_tracing() -> bool:
    """Set up Phoenix OTLP tracing. Returns True if successful."""
    global _configured
    if _configured:
        return True

    if os.environ.get("PHOENIX_TRACING_ENABLED", "true").lower() == "false":
        log.info("Phoenix tracing disabled via PHOENIX_TRACING_ENABLED=false")
        return False

    project_name = os.environ.get("PHOENIX_PROJECT_NAME", "AIML-Discovery")
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")

    try:
        from phoenix.otel import register  # type: ignore[import]
        from openinference.instrumentation.openai import OpenAIInstrumentor  # type: ignore[import]

        register(
            project_name=project_name,
            endpoint=endpoint,
        )
        OpenAIInstrumentor().instrument()
        _configured = True
        log.info(
            "Phoenix tracing active | project=%s endpoint=%s",
            project_name, endpoint,
        )
        return True
    except ImportError as exc:
        log.warning(
            "Phoenix tracing packages not installed (%s) — "
            "install arize-phoenix-otel and openinference-instrumentation-openai to enable.",
            exc,
        )
    except Exception as exc:
        log.warning(
            "Phoenix tracing failed to initialise (project=%s endpoint=%s): %s "
            "— is the Phoenix server running?",
            project_name, endpoint, exc,
        )
    return False
