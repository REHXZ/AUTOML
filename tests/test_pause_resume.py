"""Tests for the pause/stop/resume flow in job_manager.

Covers two bugs discovered when clicking pause:

Bug 1 — DEADLOCK (root cause of "stuck at pausing" forever)
============================================================
In _drive_job, when should_stop=True and a step has just been yielded,
the code enters `with job.lock:`, checks should_stop, then calls
_finish_job(job, STATUS_IDLE).  But _finish_job also uses `with job.lock:`.
threading.Lock is non-reentrant — the same thread tries to acquire a lock
it already holds → the worker thread blocks forever.

Observable symptom: job.worker.is_alive() stays True, SSE keeps heartbeating,
UI stays frozen at "pausing" and cannot be steered or reprompted.

Bug 2 — WRONG STATUS after stop between LLM iterations
=======================================================
When StopHook fires ABORT at the top of the scientist loop (between LLM
calls), no more steps are yielded.  The generator ends via StopIteration
and _drive_job calls _finish_job(job, STATUS_COMPLETE) instead of STATUS_IDLE.
This causes the session to appear "complete" rather than "idle/stopped",
which means the /messages and /stop endpoints report an unexpected state.
"""

from __future__ import annotations

import threading
from typing import Generator
from unittest.mock import MagicMock

import pytest

from backend.logic.agents.base import AutopilotStep
from backend.server.job_manager import (
    STATUS_COMPLETE,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_WAITING,
    AutopilotJob,
    _drive_job,
    _finish_job,
    _handle_step,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_DRIVE_TIMEOUT = 3.0  # seconds — generous to avoid false positives on slow CI


def _make_job(generator) -> AutopilotJob:
    """Create a minimal AutopilotJob wired to a fake generator."""
    autopilot = MagicMock()
    autopilot.set_status = MagicMock()
    job = AutopilotJob(
        project_id="test-proj",
        session_id="test-sess",
        autopilot=autopilot,
        generator=generator,
        status=STATUS_RUNNING,
    )
    return job


def _run_in_thread(job: AutopilotJob, answers=None, timeout=_DRIVE_TIMEOUT):
    """Run _drive_job in a daemon thread; return the thread for join/alive checks."""
    t = threading.Thread(target=_drive_job, args=(job, answers), daemon=True)
    t.start()
    t.join(timeout=timeout)
    return t


def _thought_step(index=1) -> AutopilotStep:
    return AutopilotStep(
        index=index, kind="thought", title="test thought",
        detail="", agent="scientist", phase="modeling",
    )


def _ask_step(index=1) -> AutopilotStep:
    return AutopilotStep(
        index=index, kind="ask", title="question",
        detail="", agent="scientist", phase="modeling",
        data={"questions": [{"question": "proceed?", "recommendation": "yes"}]},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bug 1: Deadlock when should_stop=True and a step is yielded
# ──────────────────────────────────────────────────────────────────────────────


class TestDeadlockOnStop:
    """
    Reproduce the deadlock in _drive_job's should_stop guard:

        with job.lock:          # <── lock held by this thread
            if job.should_stop:
                _finish_job(...)  # <── _finish_job ALSO does `with job.lock:` → DEADLOCK

    threading.Lock is non-reentrant.  Calling _finish_job while already holding
    job.lock causes the same thread to block waiting for itself → hangs forever.
    """

    def test_drive_job_does_not_deadlock_when_stop_set_and_step_yielded(self):
        """
        Worker must terminate within timeout even when should_stop=True
        and a non-ask step is yielded (the exact condition that triggers
        the deadlock in the buggy code).
        """
        def gen():
            yield _thought_step(1)

        job = _make_job(gen())
        job.should_stop = True  # stop was requested before/during this step

        t = _run_in_thread(job)

        assert not t.is_alive(), (
            "DEADLOCK: _drive_job is still running after "
            f"{_DRIVE_TIMEOUT}s.  The should_stop guard calls "
            "_finish_job while holding job.lock; _finish_job also "
            "uses `with job.lock:` — same thread, non-reentrant lock → "
            "deadlock.  Fix: check should_stop inside the lock but call "
            "_finish_job OUTSIDE the lock."
        )

    def test_drive_job_status_is_idle_after_stop_with_step(self):
        """After the deadlock is fixed the job status must be STATUS_IDLE."""
        def gen():
            yield _thought_step(1)

        job = _make_job(gen())
        job.should_stop = True

        _run_in_thread(job)

        assert job.status == STATUS_IDLE, (
            f"Expected STATUS_IDLE after stop, got {job.status!r}.  "
            "The worker set the wrong status."
        )

    def test_drive_job_does_not_deadlock_multiple_steps(self):
        """Deadlock is triggered after ANY yielded step, not just the first."""
        def gen():
            for i in range(5):
                yield _thought_step(i + 1)

        job = _make_job(gen())
        job.should_stop = True

        t = _run_in_thread(job)
        assert not t.is_alive(), "Deadlock with multiple steps in generator"

    def test_drive_job_does_not_deadlock_in_answers_path(self):
        """
        The same deadlock exists in the answers= path of _drive_job
        (the `if answers is not None:` branch also has `with job.lock:` →
        `_finish_job`).
        """
        answers_received = []

        def gen():
            # First send() resumes with answers; yield one step after.
            received = yield _ask_step(1)
            answers_received.extend(received or [])
            yield _thought_step(2)

        job = _make_job(gen())
        # Prime the generator to the first yield point.
        first_step = next(job.generator)
        assert first_step.kind == "ask"
        job.status = STATUS_WAITING
        job.should_stop = True

        t = _run_in_thread(job, answers=["yes"])
        assert not t.is_alive(), (
            "Deadlock in the answers= path of _drive_job"
        )
        assert job.status == STATUS_IDLE


# ──────────────────────────────────────────────────────────────────────────────
# Bug 2: Wrong status when stop fires between LLM iterations (no step yielded)
# ──────────────────────────────────────────────────────────────────────────────


class TestWrongStatusOnEarlyStop:
    """
    When StopHook aborts the scientist loop at the start of an iteration,
    no more steps are yielded.  The generator ends (StopIteration) and
    _drive_job calls _finish_job(STATUS_COMPLETE) instead of STATUS_IDLE.

    The session then looks "complete" even though it was manually stopped,
    which prevents /messages and /stop from behaving correctly.
    """

    def test_status_is_idle_not_complete_when_stop_fired_between_iterations(self):
        """
        Simulate the StopHook ABORT: generator terminates without yielding
        another step after stop was requested.
        The job status must be STATUS_IDLE, not STATUS_COMPLETE.
        """
        def gen_aborted_by_stop():
            # The scientist loop broke immediately (StopHook fired ABORT).
            # No steps yielded after stop was set.
            return
            yield  # noqa: unreachable — makes this a generator

        job = _make_job(gen_aborted_by_stop())
        job.should_stop = True

        t = _run_in_thread(job)
        assert not t.is_alive(), "Thread did not finish"

        assert job.status == STATUS_IDLE, (
            f"BUG: Expected STATUS_IDLE after stop, got {job.status!r}.  "
            "When StopHook aborts the loop between LLM iterations the "
            "generator ends via StopIteration and _drive_job incorrectly "
            "calls _finish_job(STATUS_COMPLETE).  The fix is to check "
            "job.should_stop before deciding which status to pass to "
            "_finish_job."
        )

    def test_status_is_complete_for_genuine_finish(self):
        """
        If the run genuinely finishes (no stop requested), status must still
        be STATUS_COMPLETE — we must not break the happy path.
        """
        def gen_complete():
            yield _thought_step(1)
            return
            yield  # noqa: unreachable

        job = _make_job(gen_complete())
        # should_stop stays False — normal completion

        t = _run_in_thread(job)
        assert not t.is_alive(), "Thread did not finish"

        assert job.status == STATUS_COMPLETE, (
            f"Genuine completion should still set STATUS_COMPLETE, got {job.status!r}"
        )

    def test_generator_is_cleared_after_stop(self):
        """After any stop the job.generator must be None (cannot be re-driven)."""
        def gen():
            return
            yield

        job = _make_job(gen())
        job.should_stop = True

        _run_in_thread(job)
        assert job.generator is None, "generator should be None after _finish_job"

    def test_pending_step_cleared_after_stop(self):
        """pending_step must be None after stop so the UI doesn't show a stale question."""
        def gen():
            return
            yield

        job = _make_job(gen())
        job.should_stop = True
        job.pending_step = {"kind": "ask", "title": "old question"}

        _run_in_thread(job)
        assert job.pending_step is None


# ──────────────────────────────────────────────────────────────────────────────
# Regression: normal (non-stop) paths must not be affected by the fix
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalPaths:
    """Verify that fixing the stop bugs does not break the happy paths."""

    def test_ask_step_sets_waiting_status(self):
        """An 'ask' step must pause the worker with STATUS_WAITING."""
        def gen():
            yield _ask_step(1)
            yield _thought_step(2)  # should not be reached without answers

        job = _make_job(gen())

        t = _run_in_thread(job)
        assert not t.is_alive(), "Thread did not stop after ask step"
        assert job.status == STATUS_WAITING
        assert job.pending_step is not None

    def test_answers_resume_from_waiting(self):
        """Providing answers resumes a paused generator."""
        received = []

        def gen():
            answers = yield _ask_step(1)
            received.extend(answers or [])
            yield _thought_step(2)

        job = _make_job(gen())
        # Prime to the first ask pause
        first_step = next(job.generator)
        assert first_step.kind == "ask"
        job.status = STATUS_WAITING

        t = _run_in_thread(job, answers=["my answer"])
        assert not t.is_alive()
        assert received == ["my answer"]
        assert job.status == STATUS_COMPLETE

    def test_stop_after_ask_does_not_deadlock(self):
        """
        If the user clicks Stop while the session is waiting for answers
        (STATUS_WAITING), the job is not running — there is no worker thread
        to deadlock.  The /stop endpoint handles this via recover_orphaned_session.
        This test confirms _finish_job itself is safe to call without a lock.
        """
        def gen():
            yield _ask_step(1)

        job = _make_job(gen())
        # Prime to the waiting state
        next(job.generator)
        job.status = STATUS_WAITING

        # Direct call — must not block
        _finish_job(job, STATUS_IDLE)
        assert job.status == STATUS_IDLE

    def test_no_step_generator_completes_cleanly(self):
        """Empty generator finishes with STATUS_COMPLETE (should_stop=False)."""
        def gen():
            return
            yield

        job = _make_job(gen())

        t = _run_in_thread(job)
        assert not t.is_alive()
        assert job.status == STATUS_COMPLETE

    def test_error_in_generator_sets_error_status(self):
        """Exceptions from the generator must set STATUS_ERROR."""
        def gen():
            yield _thought_step(1)
            raise ValueError("boom")

        job = _make_job(gen())

        t = _run_in_thread(job)
        assert not t.is_alive()
        assert job.status == "error"
        assert "boom" in (job.error or "")
