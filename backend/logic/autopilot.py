"""AI autopilot entry point — thin wrapper around the multi-agent system."""

from __future__ import annotations

import logging
from typing import Any, Generator

from backend.logic.agents import (
    AgentContext,
    AimlScientist,
    AutopilotStep,
)
from backend.logic.agents.hook_policies import default_hook_manager
from backend.logic.agents.hooks import HookContext, HookEvent
from backend.logic.providers import ProviderConfig, build_client, provider_from_env
from backend.services.session_store import (
    LoadedSession,
    SessionWriter,
    load_session,
    new_session_id,
)
from backend.services.project_store import DatasetInfo, ProjectStore

log = logging.getLogger(__name__)


class AiAutopilot:
    """Public façade that wires the AIML Scientist into the existing UI flow."""

    def __init__(
        self,
        provider_config: ProviderConfig | None = None,
        project_id: str = "",
        store: ProjectStore | None = None,
        user_goal: str = "",
        session_id: str | None = None,
        preloaded_session: LoadedSession | None = None,
        # Legacy: api_key accepted for backward compatibility
        api_key: str | None = None,
    ) -> None:
        if provider_config is None:
            # Backward compat: build config from api_key + env vars
            cfg = provider_from_env()
            if api_key:
                cfg.api_key = api_key
            provider_config = cfg

        self._client, self._deployment = build_client(provider_config)

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

        # Wire the hook lifecycle framework.  All agents in this run share the
        # same HookManager instance via ctx.hooks.  ctx.client/deployment let
        # hook policies (SteeringHook, ModelTesterGateHook) spawn sub-agents.
        self._ctx.hooks = default_hook_manager()
        self._ctx.client = self._client
        self._ctx.deployment = self._deployment

        self._scientist = AimlScientist(self._client, self._deployment, self._ctx)

        if resumed is not None:
            self._ctx.notebook = list(resumed.notebook)
            self._ctx.new_datasets = list(resumed.new_datasets)
            self._ctx.training_runs = list(resumed.training_runs)
            last_index = max(
                (s.index for s in resumed.steps), default=0
            )
            self._ctx._step_counter = last_index
            self._scientist.load_messages(resumed.messages, resumed.strategy_summary)
            self._is_resumed = True
        else:
            self._is_resumed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_resumed(self) -> bool:
        return self._is_resumed

    def set_status(self, status: str) -> None:
        self._session_writer.set_status(status)

    def signal_stop(self) -> None:
        self._ctx.should_stop = True

    def run(self) -> Generator[AutopilotStep, list[str] | None, None]:
        try:
            yield from self._tee(self._scientist.run())
        finally:
            self._flush_snapshots()

    def continue_with(
        self, user_message: str
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        self._ctx.should_stop = False
        try:
            yield from self._tee(self._scientist.continue_with(user_message))
        finally:
            self._flush_snapshots()

    def _tee(
        self, source: Generator[AutopilotStep, list[str] | None, None]
    ) -> Generator[AutopilotStep, list[str] | None, None]:
        try:
            step = next(source)
        except StopIteration:
            return
        while True:
            # Fire STEP_EMITTED (observe-only) before persisting or forwarding.
            if self._ctx.hooks is not None:
                try:
                    hc = HookContext(
                        event=HookEvent.STEP_EMITTED,
                        agent_name=step.agent,
                        ctx=self._ctx,
                        step=step,
                    )
                    # Drain any steps hooks emit (e.g. metrics annotations).
                    for extra in self._ctx.hooks.fire(hc):
                        try:
                            self._session_writer.append_step(extra)
                        except Exception:
                            pass
                        yield extra
                except Exception as exc:
                    log.warning("AiAutopilot._tee | STEP_EMITTED hook failed: %s", exc)

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
        try:
            self._session_writer.save_notebook(self._ctx.notebook)
            self._session_writer.save_new_datasets(self._ctx.new_datasets)
            self._session_writer.save_training_runs(self._ctx.training_runs)
            status = "complete" if self._scientist.strategy_summary else "idle"
            self._session_writer.set_status(status)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("AiAutopilot._flush_snapshots | failed: %s", exc)

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
