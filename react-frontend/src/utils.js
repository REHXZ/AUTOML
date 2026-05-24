import { AGENTS, AGENT_BY_ID, FALLBACK_AGENT, PHASES, STEP_LABELS } from "./constants";

export function formatDate(value) {
  if (!value) return "Never";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 19);
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

export function statusText(status) {
  return (status ?? "unknown").replace(/_/g, " ");
}

export function maxStepIndex(steps) {
  return steps.reduce((max, step) => Math.max(max, step.index), 0);
}

export function mergeStep(steps, incoming) {
  if (steps.some((step) => step.index === incoming.index)) return steps;
  return [...steps, incoming].sort((a, b) => a.index - b.index);
}

export function normalizeQuestions(step) {
  const raw = step?.data?.questions;
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    if (typeof item === "string") return { question: item };
    const record = item;
    return {
      question: String(record.question ?? ""),
      recommendation: String(record.recommendation ?? ""),
      alternatives: Array.isArray(record.alternatives)
        ? record.alternatives.map(String)
        : [],
      explanation: String(record.explanation ?? "")
    };
  });
}

export function stepKindLabel(kind) {
  return STEP_LABELS[kind] ?? kind.replace(/_/g, " ");
}

export function isVisibleActivityStep(step) {
  return !["tool_call", "tool_result", "agent_start", "agent_end", "phase_transition"].includes(
    step.kind
  );
}

export function visibleActivitySteps(steps) {
  return steps.filter(isVisibleActivityStep);
}

export function shortDetail(step) {
  if (!step) return "No activity yet";
  const text = step.detail || step.title || "";
  return text.replace(/\s+/g, " ").slice(0, 120);
}

export function parseFigure(step) {
  const figureJson = step?.data?.figure_json;
  if (typeof figureJson !== "string") return null;
  try {
    return JSON.parse(figureJson);
  } catch {
    return null;
  }
}

export function getPhaseState(session) {
  const latestPhase = [...(session?.steps ?? [])].reverse().find((step) => step.phase)?.phase;
  const latestIndex = Math.max(
    0,
    PHASES.findIndex((phase) => phase.id === latestPhase)
  );
  return PHASES.map((phase, index) => {
    const count = session?.steps.filter((step) => step.phase === phase.id).length ?? 0;
    const state =
      count === 0 && index > latestIndex
        ? "pending"
        : index === latestIndex
          ? "active"
          : "done";
    return { ...phase, count, state };
  });
}

export function getAgentState(session) {
  const steps = session?.steps ?? [];
  const last = steps.at(-1);
  return AGENTS.map((agent) => {
    const rawAgentSteps = steps.filter((step) => step.agent === agent.id);
    const agentSteps = visibleActivitySteps(rawAgentSteps);
    const lastRawAgentStep = rawAgentSteps.at(-1);
    const lastAgentStep = agentSteps.at(-1);
    const ended = lastRawAgentStep?.kind === "agent_end";
    const active = session?.status === "running" && last?.agent === agent.id && !ended;
    const waiting =
      session?.status === "waiting_for_input" && session.pending_step?.agent === agent.id;
    const state = waiting ? "waiting" : active ? "active" : rawAgentSteps.length ? "seen" : "idle";
    return {
      ...agent,
      state,
      steps: agentSteps.length,
      charts: agentSteps.filter((step) => step.kind === "chart").length,
      lastStep: lastAgentStep
    };
  });
}

// ── New helpers for the agent run viewer ─────────────────────────────────────
export function agentFor(stepOrId) {
  const id = typeof stepOrId === "string" ? stepOrId : stepOrId?.agent;
  return AGENT_BY_ID[id] ?? FALLBACK_AGENT;
}

// Map the API's step.kind into one of the design's narrative kinds.
const DESIGN_KIND_MAP = {
  thought: "plan",
  ask: "ask_user",
  chart: "eda",
  new_dataset: "transform",
  training: "train",
  summary: "review",
  observation: "observation",
  review: "review"
};

export function designKind(step) {
  if (!step) return "observation";
  const direct = DESIGN_KIND_MAP[step.kind];
  if (direct) return direct;
  // Heuristic on agent identity for unknown kinds.
  if (step.agent === "researcher") return "research";
  if (step.agent === "eda") return "eda";
  if (step.agent === "feature_engineering") return "transform";
  if (step.agent === "modeling" || step.agent === "fine_tuning") return "train";
  if (step.agent === "review") return "review";
  if (step.agent === "scientist") return "plan";
  return "observation";
}

const DESIGN_KIND_LABELS = {
  plan: "plan",
  research: "research",
  eda: "eda",
  observation: "observation",
  ask_user: "ask · user",
  transform: "transform",
  train: "train",
  review: "review"
};
export function designKindLabel(kind) {
  return DESIGN_KIND_LABELS[kind] ?? kind;
}

export function isTerminalStatus(status) {
  return status === "idle" || status === "complete" || status === "error";
}

// Build a column index for swimlane positioning: each visible step gets a
// monotonically increasing column based on its index order.
export function buildStepLayout(steps) {
  const visible = visibleActivitySteps(steps);
  const byIndex = [...visible].sort((a, b) => a.index - b.index);
  const cols = {};
  byIndex.forEach((step, idx) => {
    cols[step.index] = idx + 1;
  });
  return { visible: byIndex, cols, count: byIndex.length };
}

// Best-effort timestamp helpers. The API does not return durations, so we
// derive minimal sortable times from step.created_at when present, otherwise
// fall back to monotonic index-based timestamps for display.
export function stepStartSecs(step, base) {
  const ts = step?.created_at;
  if (!ts || !base) return null;
  const parsed = Date.parse(ts);
  if (Number.isNaN(parsed)) return null;
  return Math.max(0, (parsed - base) / 1000);
}

export function sessionBaseTimeMs(session) {
  const first = session?.steps?.[0]?.created_at;
  if (!first) return null;
  const parsed = Date.parse(first);
  return Number.isNaN(parsed) ? null : parsed;
}

export function formatDur(s) {
  if (s == null) return "";
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function clockAt(secs, baseMs) {
  if (baseMs == null) return "";
  const t = new Date(baseMs + secs * 1000);
  const hh = String(t.getHours()).padStart(2, "0");
  const mm = String(t.getMinutes()).padStart(2, "0");
  const ss = String(t.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function isRunningStep(step, session) {
  if (!step || !session) return false;
  if (session.status !== "running") return false;
  const lastVisible = visibleActivitySteps(session.steps ?? []).at(-1);
  return lastVisible?.index === step.index;
}

export function extractMetrics(step) {
  // Training runs surface their metrics at session.training_runs; per-step
  // detail is usually markdown. We pull anything numeric out of step.data.
  const data = step?.data ?? {};
  const m = data.metrics ?? data.best_metrics ?? null;
  if (m && typeof m === "object") return m;
  return null;
}

export function inferStepTitle(step) {
  if (!step) return "";
  if (step.title) return step.title;
  return stepKindLabel(step.kind);
}

export function getSessionStats(session) {
  const steps = visibleActivitySteps(session?.steps ?? []);
  const done = steps.filter((step) => !isPendingKind(step.kind));
  const running = session?.status === "running" ? 1 : 0;
  return {
    steps: steps.length,
    done: Math.max(0, done.length - running),
    running,
    charts: steps.filter((step) => step.kind === "chart").length,
    trainingRuns: session?.training_runs?.length ?? 0,
    generatedDatasets: session?.new_datasets?.length ?? 0,
    notes: session?.notebook?.length ?? 0
  };
}

function isPendingKind(kind) {
  return kind === "ask";
}
