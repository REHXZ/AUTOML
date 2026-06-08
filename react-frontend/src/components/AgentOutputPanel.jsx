import { ChevronDown, ChevronUp, Database, ExternalLink, LineChart, Trophy, Zap } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";

import { AGENT_BY_ID } from "../constants";
import { agentFor, extractMetrics, parseFigure } from "../utils";

const DARK_BG = "#0A0E1A";
const PLOT_FONT = { color: "#7C8597", size: 9 };

function mkLayout(overrides = {}) {
  return {
    autosize: true,
    paper_bgcolor: DARK_BG,
    plot_bgcolor: DARK_BG,
    font: PLOT_FONT,
    margin: { t: 8, r: 10, b: 28, l: 34 },
    showlegend: false,
    title: undefined,
    xaxis: { gridcolor: "#232A41", zeroline: false, tickfont: { size: 8 } },
    yaxis: { gridcolor: "#232A41", zeroline: false, tickfont: { size: 8 } },
    ...overrides
  };
}

export default function AgentOutputPanel({ session, onSelectStep }) {
  const allSteps = session?.steps ?? [];

  const chartSteps = useMemo(() => allSteps.filter((s) => s.kind === "chart"), [allSteps]);
  const trainingSteps = useMemo(() => allSteps.filter((s) => s.kind === "training"), [allSteps]);
  const featureSteps = useMemo(() => allSteps.filter((s) => s.kind === "new_dataset"), [allSteps]);

  const runningAgent = session?.status === "running" ? (allSteps.at(-1)?.agent ?? null) : null;

  const tabs = useMemo(
    () =>
      [
        chartSteps.length > 0 && {
          id: "charts",
          label: "Charts",
          count: chartSteps.length,
          icon: LineChart,
          agents: ["eda"]
        },
        trainingSteps.length > 0 && {
          id: "training",
          label: "Training",
          count: trainingSteps.length,
          icon: Zap,
          agents: ["modeling", "model_tester"]
        },
        featureSteps.length > 0 && {
          id: "features",
          label: "Features",
          count: featureSteps.length,
          icon: Database,
          agents: ["feature_engineering"]
        }
      ].filter(Boolean),
    [chartSteps.length, trainingSteps.length, featureSteps.length]
  );

  const [activeTab, setActiveTab] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const prevRunningRef = useRef(null);

  // Set initial tab
  useEffect(() => {
    if (!activeTab && tabs.length > 0) setActiveTab(tabs[0].id);
  }, [tabs, activeTab]);

  // Auto-switch tab when a new relevant agent becomes active
  useEffect(() => {
    if (!runningAgent || runningAgent === prevRunningRef.current) return;
    prevRunningRef.current = runningAgent;
    const match = tabs.find((t) => t.agents.includes(runningAgent));
    if (match) {
      setActiveTab(match.id);
      setCollapsed(false);
    }
  }, [runningAgent, tabs]);

  if (tabs.length === 0) return null;

  const activeAgentRunning = tabs.find((t) => t.agents.includes(runningAgent) && t.id === activeTab);
  const activeAgentColor = activeAgentRunning
    ? (AGENT_BY_ID[activeAgentRunning.agents[0]]?.color ?? "#06D7E8")
    : null;

  return (
    <div className="aop" style={activeAgentColor ? { borderTopColor: activeAgentColor } : undefined}>
      {/* Tab bar */}
      <div className="aop__bar">
        {tabs.map((tab) => {
          const isRunning = tab.agents.includes(runningAgent);
          const color = AGENT_BY_ID[tab.agents[0]]?.color ?? "#06D7E8";
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              className={`aop__tab${isActive ? " is-active" : ""}${isRunning ? " is-running" : ""}`}
              style={isActive ? { color, borderBottomColor: color } : undefined}
              onClick={() => {
                setActiveTab(tab.id);
                if (collapsed) setCollapsed(false);
              }}
            >
              {isRunning && (
                <span className="aop__tab-pulse" style={{ background: color }} />
              )}
              <tab.icon size={11} strokeWidth={1.75} />
              {tab.label}
              <span className="aop__badge">{tab.count}</span>
            </button>
          );
        })}

        <span className="aop__spacer" />

        {runningAgent && tabs.find((t) => t.agents.includes(runningAgent)) && (
          <span className="aop__live" style={{ color: AGENT_BY_ID[runningAgent]?.color ?? "#06D7E8" }}>
            <span
              className="aop__live-dot"
              style={{ background: AGENT_BY_ID[runningAgent]?.color ?? "#06D7E8" }}
            />
            {AGENT_BY_ID[runningAgent]?.title} running
          </span>
        )}

        <button
          className="aop__collapse"
          onClick={() => setCollapsed((c) => !c)}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? <ChevronUp size={12} strokeWidth={1.75} /> : <ChevronDown size={12} strokeWidth={1.75} />}
        </button>
      </div>

      {/* Content */}
      {!collapsed && (
        <div className="aop__body">
          {activeTab === "charts" && (
            <ChartsTab steps={chartSteps} session={session} onSelectStep={onSelectStep} />
          )}
          {activeTab === "training" && (
            <TrainingTab steps={trainingSteps} session={session} onSelectStep={onSelectStep} />
          )}
          {activeTab === "features" && (
            <FeaturesTab steps={featureSteps} onSelectStep={onSelectStep} />
          )}
        </div>
      )}
    </div>
  );
}

/* ── Charts tab ── */
function ChartsTab({ steps, session, onSelectStep }) {
  if (steps.length === 0) {
    return <div className="aop__empty">EDA charts will appear here as they are created.</div>;
  }
  return (
    <div className="aop__chart-scroll">
      {steps.map((step) => (
        <ChartCard key={step.index} step={step} session={session} onSelect={onSelectStep} />
      ))}
    </div>
  );
}

function ChartCard({ step, session, onSelect }) {
  const figure = parseFigure(step);
  const agent = agentFor(step);
  const isLast =
    session?.status === "running" && session?.steps?.at(-1)?.index === step.index;
  const chartType = step.data?.chart_type ?? step.data?.analysis_type ?? "chart";

  return (
    <div
      className={`aop-chart-card${isLast ? " is-running" : ""}`}
      style={isLast ? { borderColor: agent.color, boxShadow: `0 0 0 1px ${agent.color}40` } : undefined}
      onClick={() => onSelect?.(step.index)}
      title={step.title}
    >
      <div className="aop-chart-card__top">
        <span className="aop-chart-card__type">{chartType}</span>
        <ExternalLink size={9} strokeWidth={1.75} style={{ color: "#5A6377" }} />
      </div>
      <div className="aop-chart-card__plot">
        {figure ? (
          <Plot
            data={figure.data ?? []}
            layout={{
              ...mkLayout(),
              ...(figure.layout ?? {}),
              paper_bgcolor: DARK_BG,
              plot_bgcolor: DARK_BG,
              font: PLOT_FONT,
              margin: { t: 6, r: 8, b: 26, l: 30 },
              height: 118,
              showlegend: false,
              title: undefined,
              xaxis: {
                ...(figure.layout?.xaxis ?? {}),
                gridcolor: "#232A41",
                zeroline: false,
                tickfont: { size: 7 }
              },
              yaxis: {
                ...(figure.layout?.yaxis ?? {}),
                gridcolor: "#232A41",
                zeroline: false,
                tickfont: { size: 7 }
              }
            }}
            config={{ displaylogo: false, displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%", height: "118px" }}
          />
        ) : (
          <div className="aop-chart-card__placeholder">
            <LineChart size={18} strokeWidth={1} style={{ color: "#3A4361" }} />
          </div>
        )}
      </div>
      <div className="aop-chart-card__title">{step.title}</div>
    </div>
  );
}

/* ── Training tab ── */
function TrainingTab({ steps, session, onSelectStep }) {
  if (steps.length === 0) {
    return <div className="aop__empty">Training results will appear here once the modeling agent runs.</div>;
  }

  const latestStep = steps.at(-1);
  const metrics = extractMetrics(latestStep);
  const figure = parseFigure(latestStep);
  const runs = session?.training_runs ?? [];
  const bestModel = latestStep?.data?.best_model;

  // Leaderboard rows: prefer embedded leaderboard, fall back to training_runs summary
  const leaderboardRows = useMemo(() => {
    const embedded = runs.at(-1)?.leaderboard;
    if (Array.isArray(embedded) && embedded.length > 0) {
      return embedded.filter((r) => r.auc != null).slice(0, 7);
    }
    return runs
      .map((r, i) => ({
        algo: r.best_model ?? `run ${i + 1}`,
        auc: r.best_metrics?.auc ?? r.best_metrics?.roc_auc ?? null
      }))
      .filter((r) => r.auc != null)
      .slice(0, 7);
  }, [runs]);

  return (
    <div className="aop__training">
      {/* Metrics strip */}
      <div className="aop-train__metrics">
        {bestModel && (
          <div className="aop-train__best">
            <span className="aop-train__best-label">best</span>
            <span className="aop-train__best-name">{bestModel}</span>
          </div>
        )}
        {metrics &&
          Object.entries(metrics)
            .slice(0, 5)
            .map(([k, v]) => (
              <div key={k} className="aop-train__metric">
                <span className="aop-train__metric-k">{k}</span>
                <span className="aop-train__metric-v">{Number(v).toFixed(3)}</span>
              </div>
            ))}
        {steps.length > 1 && (
          <span className="aop-train__run-count">{steps.length} run{steps.length > 1 ? "s" : ""}</span>
        )}
      </div>

      <div className="aop-train__body">
        {/* Leaderboard */}
        {leaderboardRows.length > 0 && (
          <div className="aop-train__leaderboard">
            <div className="aop-train__section-hd">
              <Trophy size={9} strokeWidth={1.75} />
              Leaderboard
            </div>
            {leaderboardRows.map((row, i) => (
              <div key={`${row.algo}-${i}`} className={`aop-lb-row${i === 0 ? " is-top" : ""}`}>
                <span className="aop-lb-row__rank">{i === 0 ? "★" : `${i + 1}`}</span>
                <span className="aop-lb-row__name">{row.algo}</span>
                <div className="aop-lb-row__bar">
                  <div
                    className={`aop-lb-row__fill${i === 0 ? " is-top" : ""}`}
                    style={{ width: `${Math.max(0, Math.min(1, row.auc)) * 100}%` }}
                  />
                </div>
                <span className={`aop-lb-row__val${i === 0 ? " is-top" : ""}`}>
                  {Number(row.auc).toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Diagnostic chart */}
        {figure && (
          <div
            className="aop-train__chart"
            onClick={() => onSelectStep?.(latestStep.index)}
            title="Click to view full details"
          >
            <Plot
              data={figure.data ?? []}
              layout={{
                ...mkLayout({ margin: { t: 12, r: 12, b: 36, l: 42 }, height: 168 }),
                ...(figure.layout ?? {}),
                paper_bgcolor: DARK_BG,
                plot_bgcolor: DARK_BG,
                font: { ...(figure.layout?.font ?? {}), color: "#7C8597", size: 9 },
                height: 168,
                showlegend: true,
                legend: { font: { size: 8 }, bgcolor: "transparent", x: 0, y: 1 },
                title: undefined
              }}
              config={{ displaylogo: false, displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%", height: "168px" }}
            />
            <div className="aop-train__chart-hint">click to expand</div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Features tab ── */
function FeaturesTab({ steps, onSelectStep }) {
  if (steps.length === 0) {
    return <div className="aop__empty">Feature engineering datasets will appear here.</div>;
  }
  return (
    <div className="aop__features">
      {steps.map((step) => {
        const data = step.data ?? {};
        const name = data.name ?? data.dataset_name ?? step.title;
        const rows = data.row_count ?? data.rows ?? null;
        const cols = data.column_count ?? data.cols ?? null;
        return (
          <div key={step.index} className="aop-feat-card" onClick={() => onSelectStep?.(step.index)}>
            <Database size={11} strokeWidth={1.75} style={{ color: "#F59E0B", flexShrink: 0 }} />
            <div className="aop-feat-card__info">
              <span className="aop-feat-card__name">{name}</span>
              {rows != null && cols != null && (
                <span className="aop-feat-card__meta">
                  {Number(rows).toLocaleString()} rows × {cols} cols
                </span>
              )}
            </div>
            <ExternalLink size={9} strokeWidth={1.75} style={{ color: "#5A6377", marginLeft: "auto" }} />
          </div>
        );
      })}
    </div>
  );
}
