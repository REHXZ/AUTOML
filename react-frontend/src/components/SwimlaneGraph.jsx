import { MessageCircleQuestion } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { AGENTS, AGENT_ORDER } from "../constants";
import { AgentDot } from "../ui";
import {
  agentFor,
  buildStepLayout,
  designKind,
  extractMetrics,
  formatDur,
  inferStepTitle,
  isRunningStep
} from "../utils";

export default function SwimlaneGraph({
  session,
  density = "comfortable",
  selectedIndex,
  onSelect,
  hoverIndex,
  onHover
}) {
  const colW = density === "compact" ? 112 : 136;
  const rowH = density === "compact" ? 64 : 76;
  const labelW = 156;
  const padX = 32;
  const padY = 18;
  const cardH = rowH - 18;

  const { visible, cols, count } = useMemo(
    () => buildStepLayout(session?.steps ?? []),
    [session?.steps]
  );

  const rowIndex = useMemo(() => {
    const map = {};
    AGENT_ORDER.forEach((id, idx) => {
      map[id] = idx;
    });
    return map;
  }, []);

  const totalCols = Math.max(count + 1, 4);
  const canvasW = padX * 2 + colW * totalCols;
  const canvasH = padY * 2 + rowH * AGENTS.length;

  const layout = useMemo(() => {
    const out = {};
    visible.forEach((step) => {
      const col = cols[step.index];
      const row = rowIndex[step.agent] ?? 0;
      const x = padX + (col - 1) * colW;
      const y = padY + row * rowH + (rowH - cardH) / 2;
      const w = colW - 16;
      out[step.index] = {
        x,
        y,
        w,
        h: cardH,
        right: x + w,
        left: x,
        cyMid: y + cardH / 2
      };
    });
    return out;
  }, [visible, cols, rowIndex, colW, rowH, cardH]);

  const runningStep = useMemo(() => {
    if (session?.status !== "running") return null;
    return visible.at(-1) ?? null;
  }, [visible, session?.status]);

  const viewportRef = useRef(null);

  useEffect(() => {
    const target = runningStep && layout[runningStep.index];
    const v = viewportRef.current;
    if (!target || !v) return;

    const stepLeft = labelW + target.x;
    const stepRight = stepLeft + target.w;
    const laneTop = target.y;
    const laneBottom = laneTop + target.h;

    const visibleLeft = v.scrollLeft + labelW;
    const visibleRight = v.scrollLeft + v.clientWidth;
    const visibleTop = v.scrollTop;
    const visibleBottom = v.scrollTop + v.clientHeight;

    let nextLeft = v.scrollLeft;
    let nextTop = v.scrollTop;

    if (stepRight > visibleRight - 24) {
      nextLeft = stepRight - v.clientWidth + 48;
    } else if (stepLeft < visibleLeft + 24) {
      nextLeft = stepLeft - labelW - 48;
    }
    if (laneBottom > visibleBottom - 16) {
      nextTop = laneBottom - v.clientHeight + 32;
    } else if (laneTop < visibleTop + 16) {
      nextTop = laneTop - 32;
    }

    nextLeft = Math.max(0, nextLeft);
    nextTop = Math.max(0, nextTop);

    if (nextLeft !== v.scrollLeft || nextTop !== v.scrollTop) {
      v.scrollTo({ left: nextLeft, top: nextTop, behavior: "smooth" });
    }
  }, [runningStep?.index, layout, labelW]);

  const nowX = runningStep
    ? (layout[runningStep.index]?.right ?? 0) + 14
    : null;

  const connectors = useMemo(() => {
    if (visible.length < 2) return [];
    const out = [];
    for (let i = 1; i < visible.length; i += 1) {
      const parent = visible[i - 1];
      const child = visible[i];
      const a = layout[parent.index];
      const b = layout[child.index];
      if (!a || !b) continue;
      const dx = Math.max(28, (b.left - a.right) * 0.55);
      const path = `M ${a.right} ${a.cyMid} C ${a.right + dx} ${a.cyMid}, ${b.left - dx} ${b.cyMid}, ${b.left} ${b.cyMid}`;
      out.push({
        id: `${parent.index}-${child.index}`,
        path,
        fromIdx: parent.index,
        toIdx: child.index,
        color: agentFor(parent).color
      });
    }
    return out;
  }, [visible, layout]);

  return (
    <div className="graph">
      <div className="graph__viewport" ref={viewportRef}>
        <div className="graph__row">
          <div
            className="graph__labels"
            style={{ height: canvasH, paddingTop: padY }}
          >
            {AGENTS.map((agent) => {
              const AgentIcon = agent.icon;
              return (
                <div
                  key={agent.id}
                  className="graph__label-row"
                  style={{ height: rowH }}
                >
                  <span
                    className="graph__label-icon"
                    style={{
                      background: `${agent.color}1F`,
                      border: `1px solid ${agent.color}55`,
                      color: agent.color
                    }}
                  >
                    <AgentIcon size={14} strokeWidth={1.75} />
                  </span>
                  <div className="graph__label-text">
                    <span className="graph__label-name">{agent.title}</span>
                    <span className="graph__label-role">{agent.role}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="graph__canvas" style={{ width: canvasW, height: canvasH }}>
          <svg
            width={canvasW}
            height={canvasH}
            style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
          >
            <defs>
              <pattern id="apGrid" width="48" height="48" patternUnits="userSpaceOnUse">
                <path d="M 48 0 L 0 0 0 48" fill="none" stroke="#161B2C" strokeWidth="0.5" opacity="0.5" />
              </pattern>
              <linearGradient id="apNowGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#06D7E8" stopOpacity="0" />
                <stop offset="50%" stopColor="#06D7E8" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#06D7E8" stopOpacity="0" />
              </linearGradient>
            </defs>
            <rect width={canvasW} height={canvasH} fill="url(#apGrid)" />

            {AGENTS.map((_, i) => (
              <line
                key={`r-${i}`}
                x1={0}
                x2={canvasW}
                y1={padY + i * rowH}
                y2={padY + i * rowH}
                stroke="#161B2C"
                strokeWidth="1"
                opacity="0.6"
              />
            ))}
            <line
              x1={0}
              x2={canvasW}
              y1={padY + AGENTS.length * rowH}
              y2={padY + AGENTS.length * rowH}
              stroke="#161B2C"
              strokeWidth="1"
              opacity="0.6"
            />

            {connectors.map((c) => {
              const isPath =
                (selectedIndex != null && (c.fromIdx === selectedIndex || c.toIdx === selectedIndex)) ||
                (hoverIndex != null && (c.fromIdx === hoverIndex || c.toIdx === hoverIndex));
              return (
                <g key={c.id}>
                  <path
                    d={c.path}
                    fill="none"
                    stroke={c.color}
                    strokeOpacity={isPath ? 0.95 : 0.35}
                    strokeWidth={isPath ? 1.75 : 1.25}
                  />
                  <ConnectorArrow path={c.path} color={c.color} active={isPath} />
                </g>
              );
            })}

            {nowX != null ? (
              <g>
                <line
                  x1={nowX}
                  x2={nowX}
                  y1={padY - 4}
                  y2={canvasH - padY + 4}
                  stroke="#06D7E8"
                  strokeWidth="1.25"
                  strokeDasharray="3 3"
                  opacity="0.55"
                >
                  <animate attributeName="stroke-dashoffset" from="0" to="-12" dur="1.4s" repeatCount="indefinite" />
                </line>
                <rect
                  x={nowX - 1.5}
                  y={padY - 4}
                  width="3"
                  height={canvasH - padY * 2 + 8}
                  fill="url(#apNowGrad)"
                  opacity="0.7"
                />
                <rect
                  x={nowX - 22}
                  y={padY - 16}
                  width="44"
                  height="14"
                  rx="2"
                  fill="#0A0E1A"
                  stroke="#06D7E8"
                  strokeOpacity="0.6"
                />
                <text
                  x={nowX}
                  y={padY - 6}
                  textAnchor="middle"
                  fontFamily="var(--font-mono)"
                  fontSize="9"
                  letterSpacing="0.12em"
                  fill="#06D7E8"
                >
                  NOW
                </text>
              </g>
            ) : null}
          </svg>

          {visible.map((step) => (
            <StepCard
              key={step.index}
              step={step}
              layout={layout[step.index]}
              selected={selectedIndex === step.index}
              hovered={hoverIndex === step.index}
              isRunning={isRunningStep(step, session)}
              onSelect={() => onSelect(step.index)}
              onMouseEnter={() => onHover(step.index)}
              onMouseLeave={() => onHover(null)}
            />
          ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function StepCard({ step, layout, selected, hovered, isRunning, onSelect, onMouseEnter, onMouseLeave }) {
  if (!layout) return null;
  const agent = agentFor(step);
  const kind = designKind(step);
  const isAsk = kind === "ask_user";
  const accent = isAsk ? "#E91E63" : agent.color;
  const metrics = extractMetrics(step);

  const border = isRunning ? `1px solid ${accent}` : "1px solid var(--ink-700)";
  const boxShadow = isRunning
    ? `0 0 0 1px ${accent}, 0 0 24px ${accent}55`
    : selected
      ? `0 0 0 1.5px ${accent}`
      : "none";

  return (
    <div
      className="step-card"
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      title={`${agent.title} · ${kind} · click for details`}
      style={{
        left: layout.x,
        top: layout.y,
        width: layout.w,
        height: layout.h,
        border,
        boxShadow,
        transform: hovered && !selected ? "translateY(-1px)" : "none"
      }}
    >
      <div className="step-card__head">
        <div className="step-card__head-left">
          {isAsk ? (
            <MessageCircleQuestion size={11} strokeWidth={2} style={{ color: accent }} />
          ) : (
            <AgentDot color={accent} size={7} glow={isRunning} pulse={isRunning} />
          )}
          <span
            className="step-card__kind"
            style={{ color: isRunning ? accent : "var(--fg-3)" }}
          >
            {isAsk ? "ASK" : agent.short}
          </span>
        </div>
        <span
          className="step-card__time"
          style={isRunning ? { color: accent } : undefined}
        >
          {isRunning ? "RUNNING" : kind.toUpperCase()}
        </span>
      </div>

      <div className="step-card__title">{inferStepTitle(step)}</div>

      {metrics?.auc != null ? (
        <div className="step-card__metric">
          <span>
            AUC <span style={{ color: "#A7F3D0" }}>{Number(metrics.auc).toFixed(3)}</span>
          </span>
        </div>
      ) : null}

      {isRunning ? (
        <div style={{ marginTop: "auto" }}>
          <div className="step-card__progress">
            <div className="step-card__progress-bar" style={{ width: "55%" }} />
          </div>
          <div className="step-card__progress-label">{formatDur(0)} live</div>
        </div>
      ) : null}
    </div>
  );
}

function ConnectorArrow({ path, color, active }) {
  const match = path.match(/C [\d.\- ]+, [\d.\- ]+, ([\d.\-]+) ([\d.\-]+)$/);
  if (!match) return null;
  const ex = parseFloat(match[1]);
  const ey = parseFloat(match[2]);
  return (
    <polygon
      points={`${ex - 5},${ey - 2.5} ${ex - 5},${ey + 2.5} ${ex},${ey}`}
      fill={color}
      opacity={active ? 0.95 : 0.5}
    />
  );
}
