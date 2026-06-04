import {
  AlertTriangle,
  ArrowRight,
  Binary,
  Brain,
  ChartLine,
  CircleDot,
  Cpu,
  Database,
  FlaskConical,
  Lightbulb,
  MessageCircleQuestion,
  ScatterChart,
  Search,
  SlidersVertical,
  Trophy,
  X
} from "lucide-react";
import { useEffect } from "react";
import Plot from "react-plotly.js";
import ReactMarkdown from "react-markdown";

import { IconBtn, Pill } from "../ui";
import {
  agentFor,
  clockAt,
  designKind,
  extractMetrics,
  formatDur,
  inferStepTitle,
  isRunningStep,
  parseFigure,
  sessionBaseTimeMs,
  stepStartSecs
} from "../utils";

/** Format token count as compact string: 4210 → "4.2k" */
function fmtTok(n) {
  if (n == null) return "—";
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/** Derive per-agent max context tokens seen across all session steps. */
function agentCtxMap(session) {
  const map = {};
  for (const s of session?.steps ?? []) {
    const ct = s.data?.context_tokens;
    if (ct != null && (map[s.agent] == null || ct > map[s.agent])) {
      map[s.agent] = ct;
    }
  }
  return map;
}

export default function DetailDrawer({ session, step, theme, onClose }) {
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!step) return null;

  const agent = agentFor(step);
  const AgentIcon = agent.icon;
  const running = isRunningStep(step, session);
  const kind = designKind(step);
  const isAsk = kind === "ask_user";
  const metrics = extractMetrics(step);
  const figure = parseFigure(step);
  const baseMs = sessionBaseTimeMs(session);
  const secs = stepStartSecs(step, baseMs) ?? 0;
  const ctxTokens = step.data?.context_tokens ?? null;
  const agentCtx = agentCtxMap(session);

  return (
    <div className="drawer">
      <div className="drawer__head">
        <span
          className="drawer__head-icon"
          style={{
            background: `${agent.color}1F`,
            border: `1px solid ${agent.color}55`,
            color: agent.color
          }}
        >
          <AgentIcon size={14} strokeWidth={1.75} />
        </span>
        <div className="drawer__head-text">
          <span className="drawer__head-name">{agent.title}</span>
          <span className="drawer__head-meta">
            #{step.index} · {kind.replace("_", " ")}
            {baseMs ? ` · ${clockAt(secs, baseMs)}` : ""}
          </span>
        </div>
        {running ? (
          <Pill tone="running" dot="running" pulse>
            running
          </Pill>
        ) : null}
        <IconBtn icon={X} label="Close" onClick={onClose} kbd="Esc" />
      </div>

      <div className="drawer__body">
        <div
          className="drawer__hero"
          style={{
            border: `1px solid ${running ? agent.color : "var(--ink-700)"}`,
            boxShadow: running ? `0 0 0 1px ${agent.color}, 0 0 28px ${agent.color}40` : "none"
          }}
        >
          <div className="drawer__hero-head">
            <span
              className="timeline__agent-icon"
              style={{
                background: `${agent.color}1A`,
                color: agent.color,
                border: `1px solid ${agent.color}40`
              }}
            >
              <AgentIcon size={11} strokeWidth={1.75} />
            </span>
            <span className="timeline__agent-name">{agent.title}</span>
            <span className="timeline__kind-label">{kind.replace("_", " · ")}</span>
            <span style={{ flex: 1 }} />
            {running ? (
              <Pill tone="running" dot="running" pulse>
                running
              </Pill>
            ) : null}
            {metrics?.auc != null && !running ? (
              <span className="timeline__metric">
                AUC <span className="accent">{Number(metrics.auc).toFixed(3)}</span>
              </span>
            ) : null}
          </div>
          <div className="drawer__hero-title">{inferStepTitle(step)}</div>
          {step.detail ? (
            <div className="drawer__hero-body">
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.5, color: "var(--ink-100)" }}>
                {extractSummaryLine(step.detail)}
              </p>
            </div>
          ) : (
            <p className="drawer__hero-empty">No detail text recorded — see contextual sections below.</p>
          )}
          <div className="kv-row">
            <Kv k="step" v={`#${step.index}`} />
            {baseMs ? <Kv k="started" v={clockAt(secs, baseMs)} /> : null}
            <Kv k="kind" v={kind.replace("_", " ")} />
            {ctxTokens != null ? <Kv k="ctx" v={`${ctxTokens.toLocaleString()} tokens`} /> : null}
            {running ? <Kv k="state" v="live" live /> : null}
          </div>
        </div>

        {step.kind === "thought" ? (
          <Section title="Reasoning" icon={Brain}>
            <div
              className="reasoning-blockquote"
              style={{ borderLeft: `2px solid ${agent.color}` }}
            >
              {step.detail || "The Scientist is orchestrating the next phase of work."}
            </div>
          </Section>
        ) : null}

        {step.kind === "chart" ? (
          <Section title="Chart" icon={ChartLine}>
            {figure ? (
              <Plot
                data={figure.data ?? []}
                layout={{
                  ...(figure.layout ?? {}),
                  autosize: true,
                  height: 320,
                  margin: { t: 28, r: 16, b: 40, l: 44, ...(figure.layout?.margin ?? {}) },
                  paper_bgcolor:
                    figure.layout?.paper_bgcolor ??
                    (theme === "light" ? "#FFFFFF" : "#0A0E1A"),
                  plot_bgcolor:
                    figure.layout?.plot_bgcolor ??
                    (theme === "light" ? "#FFFFFF" : "#0A0E1A"),
                  font: {
                    ...(figure.layout?.font ?? {}),
                    color: figure.layout?.font?.color ?? (theme === "light" ? "#0A0E1A" : "#E5E9F2"),
                    family: "var(--font-sans)"
                  }
                }}
                config={{ displaylogo: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%", height: "320px" }}
              />
            ) : (
              <p className="empty-note">This step has no inline chart data.</p>
            )}
          </Section>
        ) : null}

        {agent.id === "eda" && step.kind !== "chart" && step.kind !== "observation" ? (
          <Section title="EDA snapshot" icon={ScatterChart}>
            <Row
              icon={CircleDot}
              color={agent.color}
              text={truncateText(step.detail || "EDA step recorded.", 400)}
            />
          </Section>
        ) : null}

        {step.kind === "ask" ? (
          <Section title="Question" icon={MessageCircleQuestion}>
            <div
              style={{
                padding: "14px 16px",
                background: "rgba(233,30,99,0.06)",
                border: "1px solid rgba(233,30,99,0.30)",
                borderRadius: "var(--radius-2)"
              }}
            >
              <p
                style={{
                  fontFamily: "var(--font-serif)",
                  fontStyle: "italic",
                  fontSize: 17,
                  lineHeight: 1.4,
                  color: "var(--ink-100)",
                  margin: 0
                }}
              >
                {step.detail || "Waiting for clarification…"}
              </p>
            </div>
          </Section>
        ) : null}

        {step.kind === "training" && metrics ? (
          <Section title="Metrics" icon={Binary}>
            <div className="metrics-grid">
              {Object.entries(metrics).map(([k, v]) => (
                <MetricCard key={k} label={k} value={Number(v).toFixed(3)} />
              ))}
            </div>
          </Section>
        ) : null}

        {step.kind === "training" && session?.training_runs?.length ? (
          <Section title="Leaderboard" icon={Trophy}>
            <Leaderboard runs={session.training_runs} />
          </Section>
        ) : null}

        {step.kind === "new_dataset" ? (
          <Section title="Generated dataset" icon={Database}>
            <div className="dataset-callout">
              <Database size={12} strokeWidth={1.75} />
              {extractDatasetSummary(step, session)}
            </div>
          </Section>
        ) : null}

        {agent.id === "model_tester" && step.data?.test_metrics ? (
          <Section title="Test Metrics (Held-Out)" icon={FlaskConical}>
            <div className="metrics-grid">
              {Object.entries(step.data.test_metrics).map(([k, v]) => (
                <MetricCard key={k} label={k} value={Number(v).toFixed(4)} />
              ))}
            </div>
            {step.data.test_size != null ? (
              <div style={{ fontSize: 11, color: "var(--fg-3)", marginTop: 6 }}>
                {step.data.test_size} test rows · {step.data.task_type} · {step.data.best_model}
              </div>
            ) : null}
          </Section>
        ) : null}

        {step.kind === "review" && step.detail ? (
          <Section title="Critique" icon={Lightbulb}>
            <StructuredDetail text={step.detail} accentColor="#F472B6" icon={AlertTriangle} />
          </Section>
        ) : null}

        {step.kind === "observation" && step.detail ? (
          <Section title="Observation" icon={CircleDot}>
            <StructuredDetail text={step.detail} accentColor="#06D7E8" icon={CircleDot} />
          </Section>
        ) : null}

        {agent.id === "researcher" && step.detail ? (
          <Section title="Findings" icon={Search}>
            {bulletize(step.detail).map((line, idx) => (
              <Row key={idx} icon={CircleDot} color="#06D7E8" text={line} />
            ))}
          </Section>
        ) : null}

        {step.kind === "summary" && step.detail ? (
          <Section title="Summary" icon={ScatterChart}>
            <div className="markdown" style={{ color: "var(--fg-2)", fontSize: 13, lineHeight: 1.55 }}>
              <ReactMarkdown>{step.detail}</ReactMarkdown>
            </div>
          </Section>
        ) : null}

        {step.kind === "thought" && step.detail ? (
          <Section title="Plan" icon={SlidersVertical}>
            <div className="markdown" style={{ color: "var(--fg-2)", fontSize: 13, lineHeight: 1.55 }}>
              <ReactMarkdown>{step.detail}</ReactMarkdown>
            </div>
          </Section>
        ) : null}

        {/* Context Usage — shown whenever any token data is available */}
        {(ctxTokens != null || Object.keys(agentCtx).length > 0) ? (
          <Section title="Context Usage" icon={Cpu}>
            {ctxTokens != null ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 10,
                  padding: "8px 12px",
                  background: "rgba(6,215,232,0.06)",
                  border: "1px solid rgba(6,215,232,0.20)",
                  borderRadius: "var(--radius-2)",
                  fontSize: 12,
                  color: "var(--ink-100)",
                }}
              >
                <Cpu size={11} strokeWidth={1.75} style={{ color: "#06D7E8", flexShrink: 0 }} />
                <span>
                  This step sent{" "}
                  <span style={{ color: "#06D7E8", fontVariantNumeric: "tabular-nums" }}>
                    {ctxTokens.toLocaleString()}
                  </span>{" "}
                  prompt tokens to the LLM
                </span>
              </div>
            ) : null}
            {Object.keys(agentCtx).length > 0 ? (
              <div>
                <div style={{ fontSize: 11, color: "var(--fg-3)", marginBottom: 6 }}>
                  Peak context size per agent (this session)
                </div>
                <div className="ctx-table">
                  {Object.entries(agentCtx)
                    .sort(([, a], [, b]) => b - a)
                    .map(([agentName, tokens]) => (
                      <div key={agentName} className="ctx-table__row">
                        <span className="ctx-table__agent">{agentName}</span>
                        <div className="ctx-table__bar-wrap">
                          <div
                            className="ctx-table__bar"
                            style={{
                              width: `${Math.round(
                                (tokens / Math.max(...Object.values(agentCtx))) * 100
                              )}%`
                            }}
                          />
                        </div>
                        <span className="ctx-table__val">{fmtTok(tokens)}</span>
                      </div>
                    ))}
                </div>
              </div>
            ) : null}
          </Section>
        ) : null}
      </div>
    </div>
  );
}

function Section({ title, icon: IconCmp, children }) {
  return (
    <div>
      <div className="section__title">
        {IconCmp ? (
          <IconCmp size={11} strokeWidth={1.75} style={{ color: "var(--fg-3)" }} />
        ) : null}
        <span className="section__title-text">{title}</span>
      </div>
      <div className="section__body">{children}</div>
    </div>
  );
}

function Kv({ k, v, live = false }) {
  return (
    <div className="kv">
      <span className="kv__k">{k}</span>
      <span className={`kv__v${live ? " is-live" : ""}`}>{v}</span>
    </div>
  );
}

function Row({ icon: IconCmp, color, text, prefix }) {
  return (
    <div className="row-item" style={{ borderLeft: `2px solid ${color}40` }}>
      {prefix ? (
        <span className="row-item__prefix" style={{ color }}>{prefix}</span>
      ) : null}
      {!prefix && IconCmp ? (
        <IconCmp size={11} strokeWidth={2} style={{ color, marginTop: 3, flexShrink: 0 }} />
      ) : null}
      <span className="row-item__text">{text}</span>
    </div>
  );
}

function MetricCard({ label, value, delta }) {
  return (
    <div className="metric-card">
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
      {delta ? (
        <div className={`metric-card__delta ${delta.startsWith("+") ? "is-up" : "is-down"}`}>
          {delta}
        </div>
      ) : null}
    </div>
  );
}

function Leaderboard({ runs }) {
  const rows = runs.slice(-1)[0]?.leaderboard ?? runs.map((run, idx) => ({
    algo: run.best_model ?? `run ${idx + 1}`,
    auc: run.best_metrics?.auc ?? run.best_metrics?.roc_auc ?? null
  }));
  const usable = rows.filter((row) => row.auc != null);
  if (usable.length === 0) return <p className="empty-note">No leaderboard yet.</p>;
  return (
    <div className="leaderboard">
      {usable.map((row, i) => (
        <div key={`${row.algo}-${i}`} className={`leaderboard__row${i === 0 ? " is-top" : ""}`}>
          <span className={`leaderboard__rank${i === 0 ? " is-top" : ""}`}>
            {i === 0 ? "★" : `${i + 1}.`}
          </span>
          <span className="leaderboard__name">{row.algo}</span>
          <div className="bar-metric">
            <div className="bar-metric__bar">
              <div
                className={`bar-metric__fill${i === 0 ? " is-top" : ""}`}
                style={{ width: `${Math.max(0, Math.min(1, row.auc)) * 100}%` }}
              />
            </div>
            <span className={`bar-metric__val${i === 0 ? " is-top" : ""}`}>
              {Number(row.auc).toFixed(3)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Extract the SUMMARY: line, or fall back to the first sentence. */
function extractSummaryLine(text) {
  const raw = String(text);
  const m = raw.match(/^SUMMARY:\s*(.+)/im);
  if (m) return m[1].trim();
  const stripped = raw.replace(/`/g, "").replace(/\*\*/g, "").replace(/\s+/g, " ").trim();
  const cut = stripped.search(/[.!?]\s|\n/);
  if (cut > 0 && cut < 250) return stripped.slice(0, cut + 1).trim();
  return stripped.length > 200 ? `${stripped.slice(0, 200).replace(/\s\S+$/, "")}…` : stripped;
}

/** Parse SUMMARY/What/Why/Detail sections from structured observation text. */
function parseStructured(text) {
  const raw = String(text);
  const get = (key) => {
    const m = raw.match(new RegExp(`^${key}:\\s*(.+)`, "im"));
    return m ? m[1].trim() : null;
  };
  const detailMatch = raw.match(/^Detail:\s*([\s\S]+)/im);
  return {
    summary: get("SUMMARY"),
    what: get("What"),
    why: get("Why"),
    detail: detailMatch ? detailMatch[1].trim() : null,
  };
}

/** Renders a structured observation/critique with What/Why rows and a collapsible Detail. */
function StructuredDetail({ text, accentColor, icon: IconCmp }) {
  const { summary, what, why, detail } = parseStructured(text);
  const hasStructure = summary || what || why;
  return (
    <div>
      {hasStructure ? (
        <>
          {summary && (
            <div style={{
              marginBottom: 8,
              padding: "8px 12px",
              background: `${accentColor}12`,
              border: `1px solid ${accentColor}30`,
              borderRadius: "var(--radius-2)",
              fontSize: 13,
              color: "var(--ink-100)",
              lineHeight: 1.5,
            }}>
              {summary}
            </div>
          )}
          {what && <Row icon={IconCmp} color={accentColor} prefix="What" text={what} />}
          {why && <Row icon={ArrowRight} color={accentColor} prefix="Why" text={why} />}
        </>
      ) : (
        <Row icon={IconCmp} color={accentColor} text={truncateText(text, 400)} />
      )}
      {detail && (
        <details style={{ marginTop: 8 }}>
          <summary style={{ cursor: "pointer", fontSize: 11, color: "var(--fg-3)", userSelect: "none" }}>
            Show full analysis
          </summary>
          <div className="markdown" style={{ marginTop: 6, fontSize: 12, color: "var(--fg-2)", lineHeight: 1.55 }}>
            <ReactMarkdown>{detail}</ReactMarkdown>
          </div>
        </details>
      )}
    </div>
  );
}

function truncateText(text, max) {
  const cleaned = String(text);
  return cleaned.length > max ? `${cleaned.slice(0, max)}…` : cleaned;
}

function bulletize(text) {
  return String(text)
    .split(/\n\s*[-*]\s*|\n\d+\.\s+|\n{2,}/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 6);
}

function extractDatasetSummary(step, session) {
  const data = step?.data ?? {};
  const name = data.name ?? data.dataset_name ?? "new dataset";
  const rows = data.row_count ?? data.rows ?? null;
  const cols = data.column_count ?? data.cols ?? null;
  if (rows && cols) return `${name} · ${Number(rows).toLocaleString()} × ${cols}`;
  const latest = session?.new_datasets?.at(-1);
  if (latest) return `${latest.name} · ${Number(latest.row_count).toLocaleString()} rows`;
  return name;
}
