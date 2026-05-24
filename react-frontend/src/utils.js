import { AGENTS, PHASES, STEP_LABELS } from "./constants";
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
  return !["tool_call", "tool_result"].includes(step.kind);
}

export function visibleActivitySteps(steps) {
  return steps.filter(isVisibleActivityStep);
}

export function shortDetail(step) {
  if (!step) return "No activity yet";
  const text = step.detail || step.title;
  return text.replace(/\s+/g, " ").slice(0, 120);
}

export function parseFigure(step) {
  const figureJson = step.data?.figure_json;
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
    const state = count === 0 && index > latestIndex ? "pending" : index === latestIndex ? "active" : "done";
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
    const waiting = session?.status === "waiting_for_input" && session.pending_step?.agent === agent.id;
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
