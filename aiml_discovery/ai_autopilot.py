"""AI autopilot entry point — thin wrapper around the multi-agent system.

The heavy lifting lives in `aiml_discovery.agents`:
  AimlScientist (orchestrator) → EdaAgent, FeatureEngineeringAgent,
  ModelingAgent, ReviewAgent, FineTuningAgent.

This module exposes the `AiAutopilot` façade the UI talks to. It also wires
each run to a `SessionWriter` so steps, LLM messages, the notebook, and the
strategy summary are streamed to disk and recoverable after a refresh.
"""

from __future__ import annotations

import logging
from typing import Any, Generator

from .agents import (
    AgentContext,
    AimlScientist,
    AutopilotStep,
    build_azure_client,
    get_deployment,
)
from .session_store import (
    LoadedSession,
    SessionWriter,
    load_session,
    new_session_id,
)
from .storage import DatasetInfo, ProjectStore

log = logging.getLogger(__name__)


class AiAutopilot:
    """Public façade that wires the AIML Scientist into the existing UI flow.

    A new instance either *starts* a fresh run (no ``session_id`` passed) or
    *resumes* an existing one (``session_id`` of a session already on disk).
    Resumed runs reload their LLM history, notebook, strategy summary, and the
    datasets/training-runs already produced — so ``continue_with`` can pick up
    seamlessly after a refresh.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        store: ProjectStore,
        user_goal: str = "",
        session_id: str | None = None,
        preloaded_session: LoadedSession | None = None,
    ) -> None:
        self._client = build_azure_client(api_key)
        self._deployment = get_deployment()

        resumed: LoadedSession | None = preloaded_session
        if session_id is not None and resumed is None:
            try:
                resumed = load_session(store, project_id, session_id)
            except FileNotFoundError:
                resumed = None

        effective_session_id = (
            session_id if resumed is not None else (session_id or new_session_id(user_goal))
        )
        effective_goal = resumed.user_goal if resumed is not None else user_goal

        self._ctx = AgentContext(
            project_id=project_id,
            store=store,
            user_goal=effective_goal,
        )
        self._session_writer = SessionWriter(
            store=store,
            project_id=project_id,
            session_id=effective_session_id,
            user_goal=effective_goal,
        )
        self._ctx.session = self._session_writer
        self._session_id = effective_session_id

        self._scientist = AimlScientist(self._client, self._deployment, self._ctx)

        if resumed is not None:
            # Rehydrate in-memory state so continue_with() can pick up.
            self._ctx.notebook = list(resumed.notebook)
            self._ctx.new_datasets = list(resumed.new_datasets)
            self._ctx.training_runs = list(resumed.training_runs)
            # Step counter must continue past the last persisted index so new
            # steps don't collide with old ones in the UI.
            last_index = max(
                (s.index for s in resumed.steps), default=0
            )
            self._ctx._step_counter = last_index
            self._scientist.load_messages(resumed.messages, resumed.strategy_summary)
            self._is_resumed = True
        else:
            self._is_resumed = False

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_resumed(self) -> bool:
        return self._is_resumed

    def set_status(self, status: str) -> None:
        """Persist a session status update for API/background runners."""
        self._session_writer.set_status(status)

    # ------------------------------------------------------------------
    # Generator surfaces used by app.py
    # ------------------------------------------------------------------

    def run(self) -> Generator[AutopilotStep, list[str] | None, None]:
        """Start a fresh discovery run, streaming AutopilotStep events."""
        try:
            yield from self._tee(self._scientist.run())
        finally:
            self._flush_snapshots()

    def continue_with(
        self, user_message: str
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        """Resume the conversation with a new user message after run() finished."""
        try:
            yield from self._tee(self._scientist.continue_with(user_message))
        finally:
            self._flush_snapshots()

    # ------------------------------------------------------------------
    # Generator boundary — persist each yielded step before passing it on
    # ------------------------------------------------------------------

    def _tee(
        self, source: Generator[AutopilotStep, list[str] | None, None]
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        """Mirror every yielded step to disk while preserving send() semantics."""
        try:
            step = next(source)
        except StopIteration:
            return
        while True:
            try:
                self._session_writer.append_step(step)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("AiAutopilot._tee | persist failed: %s", exc)
            received = yield step
            try:
                step = source.send(received)
            except StopIteration:
                return

    def _flush_snapshots(self) -> None:
        """Persist the final snapshots of notebook + outputs after the loop ends."""
        try:
            self._session_writer.save_notebook(self._ctx.notebook)
            self._session_writer.save_new_datasets(self._ctx.new_datasets)
            self._session_writer.save_training_runs(self._ctx.training_runs)
            status = "complete" if self._scientist.strategy_summary else "idle"
            self._session_writer.set_status(status)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("AiAutopilot._flush_snapshots | failed: %s", exc)

    # ------------------------------------------------------------------
    # Properties the UI reads after the run completes
    # ------------------------------------------------------------------

    @property
    def new_datasets(self) -> list[DatasetInfo]:
        return self._ctx.new_datasets

    @property
    def training_runs(self) -> list[dict[str, Any]]:
        return self._ctx.training_runs

    @property
    def strategy_summary(self) -> str:
        return self._scientist.strategy_summary

    @property
    def notebook(self) -> list[str]:
        return self._ctx.notebook


__all__ = ["AiAutopilot", "AutopilotStep"]
