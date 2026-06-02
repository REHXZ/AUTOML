import { Check, MessageCircleQuestion } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { Pill } from "../ui";
import {
  agentFor,
  clockAt,
  designKind,
  designKindLabel,
  extractMetrics,
  inferStepTitle,
  isRunningStep,
  sessionBaseTimeMs,
  stepStartSecs,
  visibleActivitySteps
} from "../utils";

export default function LinearTimeline({
  session,
  density = "comfortable",
  selectedIndex,
  onSelect,
  hoverIndex,
  onHover
}) {
  const dense = density === "compact";
  const containerRef = useRef(null);
  const runningRef = useRef(null);

  const steps = useMemo(
    () => visibleActivitySteps(session?.steps ?? []),
    [session?.steps]
  );

  const baseMs = useMemo(() => sessionBaseTimeMs(session), [session]);

  useEffect(() => {
    if (runningRef.current && containerRef.current) {
      const c = containerRef.current;
      const r = runningRef.current;
      const offset = r.offsetTop - c.clientHeight + r.clientHeight + 60;
      c.scrollTop = Math.max(0, offset);
    }
  }, [session?.session_id]);

  if (steps.length === 0) {
    return (
      <div ref={containerRef} className="timeline">
        <div className="timeline__inner">
          <p className="empty-note">Open or start a session to see activity.</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="timeline">
      <div className="timeline__inner">
        <div className="timeline__axis" />
        {steps.map((step) => {
          const agent = agentFor(step);
          const running = isRunningStep(step, session);
          const kind = designKind(step);
          const isAsk = kind === "ask_user";
          const isSel = selectedIndex === step.index;
          const isHov = hoverIndex === step.index;
          const metrics = extractMetrics(step);
          const secs = stepStartSecs(step, baseMs) ?? 0;

          return (
            <div
              key={step.index}
              ref={running ? runningRef : null}
              onClick={() => onSelect(step.index)}
              onMouseEnter={() => onHover(step.index)}
              onMouseLeave={() => onHover(null)}
              className="timeline__row"
              style={{ paddingBottom: dense ? 12 : 0 }}
            >
              <div className="timeline__time">
                <div
                  className={`timeline__time-clock${running ? " is-running" : ""}`}
                >
                  {clockAt(secs, baseMs) || `#${step.index}`}
                </div>
                <div className="timeline__time-dur">{designKindLabel(kind)}</div>
              </div>

              <div className="timeline__dot-col">
                {isAsk ? (
                  <span className="timeline__dot is-ask">
                    <MessageCircleQuestion size={9} strokeWidth={2} style={{ color: "#E91E63" }} />
                  </span>
                ) : (
                  <span
                    className={`timeline__dot${running ? " is-running" : ""}`}
                    style={{
                      background: agent.color,
                      boxShadow: running
                        ? `0 0 0 2px ${agent.color}, 0 0 16px ${agent.color}`
                        : "none"
                    }}
                  />
                )}
              </div>

              <div className="timeline__card-wrap">
                <div
                  className={`timeline__card${isSel ? " is-selected" : ""}${isHov && !isSel ? " is-hover" : ""}`}
                  style={{
                    border:
                      "1px solid " +
                      (isSel ? agent.color : running ? agent.color : "var(--ink-700)"),
                    boxShadow: running
                      ? `0 0 0 1px ${agent.color}, 0 0 28px ${agent.color}40`
                      : isSel
                        ? `0 0 0 1px ${agent.color}`
                        : "none"
                  }}
                >
                  <div className="timeline__card-head">
                    <span
                      className="timeline__agent-icon"
                      style={{
                        background: `${agent.color}1A`,
                        color: agent.color,
                        border: `1px solid ${agent.color}40`
                      }}
                    >
                      <agent.icon size={11} strokeWidth={1.75} />
                    </span>
                    <span className="timeline__agent-name">{agent.title}</span>
                    <span className="timeline__kind-label">{designKindLabel(kind)}</span>
                    <span style={{ flex: 1 }} />
                    {running ? (
                      <Pill tone="running" dot="running" pulse>
                        running
                      </Pill>
                    ) : null}
                    {isAsk && session?.status !== "waiting_for_input" ? (
                      <Pill tone="magenta" icon={Check}>
                        answered
                      </Pill>
                    ) : null}
                    {metrics?.auc != null && !running ? (
                      <span className="timeline__metric">
                        AUC{" "}
                        <span className="accent">
                          {Number(metrics.auc).toFixed(3)}
                        </span>
                      </span>
                    ) : null}
                  </div>

                  <div className="timeline__title">{inferStepTitle(step)}</div>
                  {step.detail ? (
                    <div className="timeline__summary">
                      {truncateDetail(step.detail)}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}

        <div className="timeline__live">
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: 999,
              background: "#06D7E8",
              boxShadow: "0 0 10px #06D7E8",
              animation: "apPulse 1.6s ease-in-out infinite"
            }}
          />
          live · {session?.status ?? "idle"}
        </div>
      </div>
    </div>
  );
}

function truncateDetail(text) {
  const cleaned = String(text).replace(/\s+/g, " ").trim();
  return cleaned.length > 260 ? `${cleaned.slice(0, 260)}…` : cleaned;
}
