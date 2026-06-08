import { useEffect, useRef, useState } from "react";

/* ── SVG canvas ─────────────────────────────────────── */
const W = 560;
const H = 440;
const CX = W / 2;   // 280
const CY = H / 2;   // 220
const R = 148;

const rad = (d) => (d * Math.PI) / 180;
const orbit = (angle) => ({
  x: +(CX + R * Math.cos(rad(angle))).toFixed(2),
  y: +(CY - R * Math.sin(rad(angle))).toFixed(2),
});

/* Six specialist agents arranged in a hexagon */
const AGENTS = [
  { id: "eda",      label: "EDA",          sub: "Data Profiling",    angle: 90   },
  { id: "feat",     label: "Feature Eng",  sub: "Transforms",        angle: 30   },
  { id: "model",    label: "Modeling",     sub: "25+ Models",        angle: -30  },
  { id: "finetune", label: "Fine Tuning",  sub: "Optuna HPO",        angle: -90  },
  { id: "review",   label: "Review",       sub: "QA & Critique",     angle: 210  },
  { id: "drift",    label: "Drift",        sub: "Observability",     angle: 150  },
].map((a) => ({ ...a, ...orbit(a.angle) }));

/* Six CRISP-DM steps with matching agent and drip-in log lines */
const STEPS = [
  {
    phase: "Business Understanding",
    agentId: null,
    logs: [
      'Scientist: task="Predict customer churn"',
      "Identifying target: churn_flag  |  type: binary",
      "Planning CRISP-DM pipeline · dispatching to EDA…",
    ],
  },
  {
    phase: "Data Understanding",
    agentId: "eda",
    logs: [
      "EdaAgent: 50,000 rows × 23 features loaded",
      "Class imbalance detected — positive rate: 14.8 %",
      "12 Plotly diagnostic charts generated",
    ],
  },
  {
    phase: "Data Preparation",
    agentId: "feat",
    logs: [
      "FeatureEngineeringAgent: scanning 50+ transforms",
      "OneHotEncoder applied to 8 categorical columns",
      "StandardScaler applied · 2 low-variance cols removed",
    ],
  },
  {
    phase: "Modeling",
    agentId: "model",
    logs: [
      "ModelingAgent: evaluating 25 AutoML candidates…",
      "Winner: GradientBoostingClassifier  ROC-AUC=0.891",
      "Artifact saved → ./output/best_model.pkl",
    ],
  },
  {
    phase: "Evaluation",
    agentId: "review",
    logs: [
      "ReviewAgent: no target leakage detected ✓",
      "Recall on minority class: 0.71 — flagged",
      "Steering → FineTuningAgent for improvement",
    ],
  },
  {
    phase: "Iteration",
    agentId: "finetune",
    logs: [
      "FineTuningAgent: Optuna search · 100 trials",
      "Best params: n_estimators=350  lr=0.042  depth=5",
      "ROC-AUC: 0.891 → 0.913  (+2.5 %)  ✓ converged",
    ],
  },
];

/* Corner L-shapes around the Scientist node ─────────── */
const SCW = 118;   // Scientist box width
const SCH = 56;    // Scientist box height
const SX = CX - SCW / 2;
const SY = CY - SCH / 2;
const CORNER_SIZE = 10;

const CORNERS = [
  // top-left
  `M${SX + CORNER_SIZE},${SY} L${SX},${SY} L${SX},${SY + CORNER_SIZE}`,
  // top-right
  `M${SX + SCW - CORNER_SIZE},${SY} L${SX + SCW},${SY} L${SX + SCW},${SY + CORNER_SIZE}`,
  // bottom-left
  `M${SX},${SY + SCH - CORNER_SIZE} L${SX},${SY + SCH} L${SX + CORNER_SIZE},${SY + SCH}`,
  // bottom-right
  `M${SX + SCW},${SY + SCH - CORNER_SIZE} L${SX + SCW},${SY + SCH} L${SX + SCW - CORNER_SIZE},${SY + SCH}`,
];

/* ── Component ──────────────────────────────────────── */
export default function AutopilotShowcase() {
  const [stepIdx, setStepIdx] = useState(0);
  const [visibleLogs, setVisibleLogs] = useState([]);
  const [logTick, setLogTick] = useState(0);
  const logRef = useRef(null);

  /* Cycle phases every 4.2 s */
  useEffect(() => {
    const id = setInterval(() => {
      setStepIdx((i) => (i + 1) % STEPS.length);
      setVisibleLogs([]);
      setLogTick(0);
    }, 4200);
    return () => clearInterval(id);
  }, []);

  /* Drip-in log lines */
  useEffect(() => {
    const step = STEPS[stepIdx];
    if (logTick >= step.logs.length) return;
    const delay = logTick === 0 ? 320 : 1000;
    const id = setTimeout(() => {
      setVisibleLogs((prev) => [...prev, step.logs[logTick]]);
      setLogTick((n) => n + 1);
    }, delay);
    return () => clearTimeout(id);
  }, [stepIdx, logTick]);

  /* Auto-scroll log */
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [visibleLogs]);

  const step = STEPS[stepIdx];
  const activeAgent = AGENTS.find((a) => a.id === step.agentId) ?? null;

  const fwdPath = activeAgent ? `M${CX},${CY} L${activeAgent.x},${activeAgent.y}` : null;
  const retPath = activeAgent ? `M${activeAgent.x},${activeAgent.y} L${CX},${CY}` : null;

  return (
    <div className="lp-ap-outer">
      {/* ── Network diagram ──────────────────────────── */}
      <div className="lp-ap-network">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="lp-ap-svg"
          aria-label="AI agent swarm network"
        >
          {/* dim static dashed spokes */}
          {AGENTS.map((a) => (
            <line
              key={`spoke-${a.id}`}
              x1={CX} y1={CY} x2={a.x} y2={a.y}
              stroke="#283535" strokeWidth="1" strokeDasharray="4 7"
            />
          ))}

          {/* active spoke highlight */}
          {activeAgent && (
            <line
              x1={CX} y1={CY} x2={activeAgent.x} y2={activeAgent.y}
              stroke="#63f7ff" strokeWidth="1.5" opacity="0.35"
            />
          )}

          {/* → delegation pulse (Scientist → Agent) */}
          {fwdPath && (
            <circle r="4" fill="#63f7ff">
              <animateMotion dur="0.88s" repeatCount="indefinite" path={fwdPath} />
              <animate attributeName="opacity" values="0.95;0.95;0.1" dur="0.88s" repeatCount="indefinite" />
            </circle>
          )}

          {/* ← result pulse (Agent → Scientist) */}
          {retPath && (
            <circle r="2.5" fill="#b9caca">
              <animateMotion dur="0.88s" begin="0.48s" repeatCount="indefinite" path={retPath} />
              <animate attributeName="opacity" values="0;0.65;0.65;0.05" dur="0.88s" begin="0.48s" repeatCount="indefinite" />
            </circle>
          )}

          {/* ── Agent nodes ──────────────────────────── */}
          {AGENTS.map((a) => {
            const active = a.id === step.agentId;
            const NW = 92, NH = 44;
            return (
              <g key={`agent-${a.id}`}>
                {/* Glow ring when active */}
                {active && (
                  <circle cx={a.x} cy={a.y} r="36" fill="none" stroke="#63f7ff" strokeWidth="1">
                    <animate attributeName="r"       values="34;48;34" dur="1.9s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.35;0;0.35" dur="1.9s" repeatCount="indefinite" />
                  </circle>
                )}
                {/* Node box */}
                <rect
                  x={a.x - NW / 2} y={a.y - NH / 2}
                  width={NW} height={NH}
                  fill={active ? "#002024" : "#0d1717"}
                  stroke={active ? "#63f7ff" : "#2e4040"}
                  strokeWidth={active ? 1.5 : 1}
                />
                {/* Agent name */}
                <text
                  x={a.x} y={a.y - 5}
                  textAnchor="middle"
                  fill={active ? "#63f7ff" : "#b9caca"}
                  fontSize="10"
                  fontFamily="'JetBrains Mono', monospace"
                  fontWeight="500"
                >
                  {a.label.toUpperCase()}
                </text>
                {/* Sub-label */}
                <text
                  x={a.x} y={a.y + 11}
                  textAnchor="middle"
                  fill={active ? "#849495" : "#3a4a4a"}
                  fontSize="8"
                  fontFamily="'JetBrains Mono', monospace"
                >
                  {a.sub}
                </text>
              </g>
            );
          })}

          {/* ── Scientist (orchestrator) ─────────────── */}
          {/* Outer glow box */}
          <rect x={SX - 4} y={SY - 4} width={SCW + 8} height={SCH + 8}
            fill="none" stroke="#63f7ff" strokeWidth="0.5" opacity="0.15" />
          {/* Main box */}
          <rect x={SX} y={SY} width={SCW} height={SCH}
            fill="#001820" stroke="#63f7ff" strokeWidth="2" />
          {/* Inner inset */}
          <rect x={SX + 3} y={SY + 3} width={SCW - 6} height={SCH - 6}
            fill="none" stroke="#63f7ff" strokeWidth="0.5" opacity="0.25" />

          {/* Scientist labels */}
          <text
            x={CX} y={CY - 7}
            textAnchor="middle"
            fill="#63f7ff"
            fontSize="11" fontFamily="'JetBrains Mono', monospace"
            fontWeight="500" letterSpacing="1.5"
          >SCIENTIST</text>
          <text
            x={CX} y={CY + 12}
            textAnchor="middle"
            fill="#849495"
            fontSize="8" fontFamily="'JetBrains Mono', monospace"
            letterSpacing="0.5"
          >ORCHESTRATOR</text>

          {/* Corner L-marks */}
          {CORNERS.map((d, i) => (
            <path key={`corner-${i}`} d={d} fill="none" stroke="#63f7ff" strokeWidth="1.5" />
          ))}

          {/* Heartbeat line at bottom of SVG */}
          <polyline
            points={`20,${H - 18} 50,${H - 18} 60,${H - 32} 70,${H - 10} 82,${H - 24} 90,${H - 18} ${W - 20},${H - 18}`}
            fill="none" stroke="#63f7ff" strokeWidth="1" opacity="0.18"
          />
        </svg>
      </div>

      {/* ── Sidebar ──────────────────────────────────── */}
      <div className="lp-ap-sidebar">
        {/* CRISP-DM phase list */}
        <div className="lp-ap-phases">
          <div className="lp-ap-block-title">CRISP-DM PIPELINE</div>
          {STEPS.map((s, i) => (
            <div
              key={s.phase}
              className={[
                "lp-ap-phase",
                i === stepIdx ? "lp-ap-phase--active" : "",
                i < stepIdx  ? "lp-ap-phase--done"   : "",
              ].filter(Boolean).join(" ")}
            >
              <span className="lp-ap-phase-num">{String(i + 1).padStart(2, "0")}</span>
              <span className="lp-ap-phase-dot" />
              <span className="lp-ap-phase-label">{s.phase}</span>
            </div>
          ))}
        </div>

        {/* Activity log */}
        <div className="lp-ap-log">
          <div className="lp-ap-block-title">AGENT OUTPUT</div>
          <div className="lp-ap-log-body" ref={logRef}>
            {visibleLogs.map((line, i) => (
              <div key={`${stepIdx}-${i}`} className="lp-ap-log-line">
                <span className="lp-ap-log-prompt">&gt;</span> {line}
              </div>
            ))}
            {logTick < STEPS[stepIdx].logs.length && (
              <span className="lp-ap-log-cursor">▋</span>
            )}
          </div>
        </div>

        {/* Live status badge */}
        <div className="lp-ap-status">
          <span className="lp-ap-status-dot" />
          <span className="lp-ap-status-phase">{step.phase}</span>
          <span className="lp-ap-status-idx">
            {String(stepIdx + 1).padStart(2, "0")} / {String(STEPS.length).padStart(2, "0")}
          </span>
        </div>
      </div>
    </div>
  );
}
