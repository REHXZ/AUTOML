import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { AGENTS } from "../constants";
import { IconBtn } from "../ui";
import ViewToggle from "./ViewToggle";

export default function RunHeader({ project, session, datasetSummary, stats, streaming, tweaks, playing, onTogglePlay, onView }) {
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
              const isRunning =
                session.status === "running" &&
                session.steps?.at(-1)?.agent === agent.id;
              return (
                <div key={agent.id} className="run-header__legend-item">
                  <span
                    className={`agent-dot${isRunning ? " agent-dot--pulse" : ""}`}
                    style={{
                      width: 7,
                      height: 7,
                      background: agent.color,
                      boxShadow: isRunning ? `0 0 12px ${agent.color}` : "none"
                    }}
                  />
                  <span
                    className={`run-header__legend-name${isRunning ? " is-running" : ""}`}
                    style={{ color: isRunning ? agent.color : "var(--fg-2)" }}
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
          <IconBtn icon={SkipBack} label="Jump to start" size={26} />
          <IconBtn
            icon={playing ? Pause : Play}
            label={playing ? "Pause" : "Play replay"}
            onClick={onTogglePlay}
            size={26}
            active={playing}
          />
          <IconBtn icon={SkipForward} label="Jump to now" size={26} />
          <div style={{ width: 1, height: 16, background: "var(--ink-700)", margin: "0 4px" }} />
          <span className="run-header__controls-speed">{tweaks.replaySpeed}</span>
        </div>

        <div className="run-header__stats">
          <em>steps</em> <strong>{stats.steps}</strong> <em>done ·</em>{" "}
          <span className="accent">{stats.running}</span> <em>running</em>
          {streaming ? <span className="accent"> · stream</span> : null}
        </div>
      </div>
    </div>
  );
}
