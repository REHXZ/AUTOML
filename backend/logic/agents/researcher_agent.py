"""Researcher Agent: searches the web via SearXNG and synthesises findings for scientists and peer agents."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Generator

from .base import AgentContext, AutopilotStep, BaseAgent

log = logging.getLogger(__name__)

_SEARXNG_BASE = "http://localhost:8080"
_DEFAULT_NUM_RESULTS = 5
_DEFAULT_MAX_CHARS = 4000

_SYSTEM_PROMPT = """\
You are the Researcher Agent — a rigorous scientific investigator with access
to the web via SearXNG.

Your job is to answer research questions by searching, reading sources, and
synthesising well-cited findings. You are called by:
  • Scientists who want to investigate a topic or uncertainty in their data.
  • The Review Agent, who wants literature context to validate model choices.
  • The Modeling Agent, who wants benchmarks or technique recommendations.

Operating principles:
  1. SEARCH FIRST. Call search_web before drawing any conclusions.
  2. FETCH THE SOURCES. When a result looks highly relevant, call fetch_page
     to read the actual content — not just the snippet. Favour primary sources
     (papers, official docs, authoritative blogs) over aggregators.
  3. CROSS-CHECK. Search multiple angles (technique name, problem type,
     dataset domain). Note disagreements between sources.
  4. STAY GROUNDED. Only report what is supported by sources you actually read.
     Quote key passages verbatim and always cite the URL.
  5. BE CONCISE. The calling agent needs actionable insight, not raw dumps.
     Lead with the headline finding, then supporting detail.
  6. FLAG UNCERTAINTY. If sources conflict or the evidence is thin, say so
     explicitly — a hedged answer is better than a confident wrong one.

Typical flow:
  search_web(broad query) → review snippets → fetch_page(best 1-2 URLs)
  → record_finding for each key insight → done(summary)

Call record_finding(text) for each key insight (include source URL).
Finish with done(summary, key_findings, sources).
"""


# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────────────────


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": (
                    "Search the web via SearXNG. Returns title, URL, and snippet "
                    "for the top results. Call this first before fetching pages."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results (default 5, max 10).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": (
                    "Fetch and read the plain-text content of a web page. "
                    "Use this on promising URLs from search results to get the full content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL of the page to fetch.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return (default 4000, max 10000).",
                            "default": 4000,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_finding",
                "description": (
                    "Write a research insight to the shared notebook. "
                    "Always include the source URL in the text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Finding text, including source URL.",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Finish research with a structured summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Narrative summary answering the research question.",
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Bullet-point list of the most important findings.",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source URLs cited in the research.",
                        },
                    },
                    "required": ["summary"],
                },
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers (stdlib only — no extra dependencies)
# ──────────────────────────────────────────────────────────────────────────────


def _search_searxng(query: str, num_results: int) -> str:
    num_results = min(max(1, num_results), 10)
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{_SEARXNG_BASE}/search?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AIML-ResearcherAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("search_web | SearXNG request failed: %s", exc)
        return json.dumps({"error": f"SearXNG search failed: {exc}"})

    results = data.get("results", [])[:num_results]
    formatted = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "engines": r.get("engines", []),
        }
        for r in results
    ]
    log.info("search_web | query=%r returned %d results", query, len(formatted))
    return json.dumps({"query": query, "num_results": len(formatted), "results": formatted})


def _fetch_page(url: str, max_chars: int) -> str:
    max_chars = min(max(500, max_chars), 10_000)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("fetch_page | request failed url=%s: %s", url, exc)
        return json.dumps({"error": f"Could not fetch page: {exc}"})

    # Strip <script>, <style> blocks, then all remaining tags, then HTML entities.
    text = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>",
        " ",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    truncated = len(text) > max_chars
    text = text[:max_chars]
    log.info("fetch_page | url=%s chars=%d truncated=%s", url, len(text), truncated)
    return json.dumps({
        "url": url,
        "content": text,
        "truncated": truncated,
        "chars_returned": len(text),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────


class ResearcherAgent(BaseAgent):
    """Web-search powered research agent. Callable by Scientist, Review, and Modeling agents."""

    name = "researcher"
    display_name = "Researcher Agent"

    def __init__(self, client, deployment: str, context: AgentContext) -> None:
        super().__init__(client, deployment, context)
        self._summary: dict[str, Any] = {}

    def run(
        self, question: str
    ) -> Generator[AutopilotStep, list[str] | None, dict[str, Any]]:
        yield self._step(
            "agent_start",
            "Researcher Agent dispatched",
            question or "(general web research)",
        )

        user_prompt = (
            f"Research question:\n{question}\n\n"
            f"Context from shared notebook:\n{self._ctx.notebook_text()}\n\n"
            f"User goal: {self._ctx.user_goal or '(none)'}\n\n"
            "Search the web, read the most relevant sources, and synthesise your findings."
        )

        yield from self.run_llm_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=_tools(),
            dispatch=self._dispatch,
            max_iterations=16,
            thought_title="Researcher Agent — Reasoning",
        )

        if self._summary:
            yield self._step(
                "observation",
                "Research Summary",
                self._summary.get("summary", ""),
                data=self._summary,
            )

        yield self._step("agent_end", "Researcher Agent finished", "")
        return self._summary or {"summary": "Researcher agent ended without summary."}

    def _dispatch(
        self, name: str, args: dict, tool_call_id: str
    ) -> tuple[str | None, AutopilotStep | None, bool]:
        if name == "search_web":
            query = (args.get("query") or "").strip()
            num = int(args.get("num_results") or _DEFAULT_NUM_RESULTS)
            if not query:
                return json.dumps({"error": "query is required"}), None, False
            result = _search_searxng(query, num)
            step = self._step("tool_result", f"Web search: {query}", f"{num} results requested")
            return result, step, False

        if name == "fetch_page":
            url = (args.get("url") or "").strip()
            max_chars = int(args.get("max_chars") or _DEFAULT_MAX_CHARS)
            if not url:
                return json.dumps({"error": "url is required"}), None, False
            result = _fetch_page(url, max_chars)
            step = self._step("tool_result", f"Fetched page: {url[:80]}", f"max {max_chars} chars")
            return result, step, False

        if name == "record_finding":
            text = (args.get("text") or "").strip()
            if text:
                self._ctx.notebook.append(f"[Research] {text}")
            return (
                json.dumps({"recorded": True}),
                self._step("observation", "Research finding", text),
                False,
            )

        if name == "done":
            self._summary = args
            return json.dumps({"status": "noted"}), None, True

        return json.dumps({"error": f"Unknown tool: {name}"}), None, False
