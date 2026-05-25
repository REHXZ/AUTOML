import { useMemo } from "react";

import { agentFor, visibleActivitySteps } from "../utils";

export default function LiveStream({ session }) {
  const lines = useMemo(() => buildStreamLines(session), [session]);

  if (lines.length === 0) {
    return (
      <div className="live-stream">
        <div className="live-stream__label">
          <span className="live-stream__label-dot" />
          <span className="live-stream__label-text">stream</span>
        </div>
        <div className="live-stream__track">
          <div className="live-stream__marquee">
            <span className="live-stream__msg" style={{ paddingLeft: 12 }}>
              waiting for agent activity…
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="live-stream">
      <div className="live-stream__label">
        <span className="live-stream__label-dot" />
        <span className="live-stream__label-text">stream</span>
      </div>
      <div className="live-stream__track">
        <div className="live-stream__marquee">
          {[...lines, ...lines].map((line, idx) => {
            const agent = agentFor(line.agent);
            return (
              <div key={idx} className="live-stream__item">
                <span className="live-stream__t">#{line.index}</span>
                <span
                  className="agent-dot"
                  style={{ width: 5, height: 5, background: agent.color }}
                />
                <span className="live-stream__agent">{agent.short}</span>
                <span className="live-stream__msg">{line.text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function buildStreamLines(session) {
  const steps = visibleActivitySteps(session?.steps ?? []);
  return steps.slice(-6).map((step) => ({
    index: step.index,
    agent: step.agent,
    text: (step.title || step.detail || step.kind).slice(0, 80)
  }));
}
