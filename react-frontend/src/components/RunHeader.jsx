import { Loader2, Pause, Play } from "lucide-react";

import { AGENTS } from "../constants";
import { IconBtn } from "../ui";
import ViewToggle from "./ViewToggle";

export default function RunHeader({ project, session, datasetSummary, stats, streaming, stopping, tweaks, onPause, onResume, onView }) {
  const isRunning = session.status === "running";
  const isPaused = session.status === "idle" && (session.steps?.length ?? 0) > 0;

  return (
    <div className="run-header">
      <div className="run-header__top">
        <div className="run-header__identity">
          <div className="run-header__name-row">
            <span className="eyebrow">autopilot session</span>
            <span className="run-header__name">{session.title || session.session_id}</span>
            <span className="run-header__dataset">{datasetSummary}</span>
          </div>
          <div className="run-header__goal">"{session.user_goal || project?.description || "Discover insight, build models, iterate."}"</div>
        </div>
        <ViewToggle value={tweaks.view} onChange={onView} />
      </div>

      <div className="run-header__row2">
        {tweaks.showLegend ? (
          <div className="run-header__legend">
            <span className="eyebrow">agents</span>
            {AGENTS.map((agent) => {
              const count =
                session.steps?.filter((step) => step.agent === agent.id).length ?? 0;
              const isAgentRunning =
                isRunning && session.steps?.at(-1)?.agent === agent.id;
              return (
                <div key={agent.id} className="run-header__legend-item">
                  <span
                    className={`agent-dot${isAgentRunning ? " agent-dot--pulse" : ""}`}
                    style={{
                      width: 7,
                      height: 7,
                      background: agent.color,
                      boxShadow: isAgentRunning ? `0 0 12px ${agent.color}` : "none"
                    }}
                  />
                  <span
                    className={`run-header__legend-name${isAgentRunning ? " is-running" : ""}`}
                    style={{ color: isAgentRunning ? agent.color : "var(--fg-2)" }}
                  >
                    {agent.title}
                  </span>
                  <span className="run-header__legend-count">×{count}</span>
                </div>
              );
            })}
          </div>
        ) : null}

        <div style={{ flex: 1 }} />

        <div className="run-header__controls">
          {isRunning ? (
            <IconBtn
              icon={stopping ? Loader2 : Pause}
              label={stopping ? "Stopping…" : "Pause run"}
              onClick={stopping ? undefined : onPause}
              size={26}
              active={!stopping}
              style={{ opacity: stopping ? 0.5 : 1, cursor: stopping ? "default" : "pointer" }}
            />
          ) : isPaused ? (
            <IconBtn
              icon={Play}
              label="Resume run"
              onClick={onResume}
              size={26}
            />
          ) : null}
        </div>

        <div className="run-header__stats">
          <em>steps</em> <strong>{stats.steps}</strong> <em>done ·</em>{" "}
          <span className="accent">{stats.running}</span> <em>running</em>
          {streaming ? <span className="accent"> · stream</span> : null}
          {stopping ? <span style={{ color: "var(--fg-2)" }}> · stopping</span> : null}
          {isPaused ? <span style={{ color: "var(--warning, #f59e0b)" }}> · paused</span> : null}
        </div>
      </div>
    </div>
  );
}
