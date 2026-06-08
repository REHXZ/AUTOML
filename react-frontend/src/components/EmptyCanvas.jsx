import { Eye, EyeOff, Loader2, Play } from "lucide-react";
import { useState } from "react";

import { Btn } from "../ui";

export default function EmptyCanvas({ project, datasets, health, goal, onGoal, apiKey, onApiKey, onStart, loading, disabledLaunch }) {
  const [showKey, setShowKey] = useState(false);
  const missingKey = !apiKey?.trim();
  const launchDisabled = disabledLaunch || missingKey;

  return (
    <div className="canvas-row">
      <div className="empty-canvas">
        <span className="eyebrow">{project ? "ready to launch" : "select a project"}</span>
        <h1 className="empty-canvas__title">
          {project?.name ?? "No project selected"}
        </h1>
        <p className="empty-canvas__hint">
          {project?.description ||
            "Pick a project from the workspace, upload a dataset, and describe what you want the agents to investigate."}
        </p>
        <div className="empty-canvas__launch">
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="eyebrow" style={{ color: "var(--fg-3)" }}>OpenAI API Key</label>
            <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
              <input
                className="field"
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => onApiKey(e.target.value)}
                placeholder="sk-…"
                style={{ flex: 1, paddingRight: 36, fontFamily: "monospace", fontSize: 13 }}
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                onClick={() => setShowKey((v) => !v)}
                style={{ position: "absolute", right: 8, background: "none", border: "none", cursor: "pointer", color: "var(--fg-3)", display: "flex", alignItems: "center" }}
                tabIndex={-1}
                aria-label={showKey ? "Hide API key" : "Show API key"}
              >
                {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            {missingKey && (
              <span style={{ fontSize: 11, color: "var(--warning, #f59e0b)" }}>
                Enter your OpenAI API key to launch a run. It is stored only in your browser.
              </span>
            )}
          </div>
          <textarea
            className="field"
            value={goal}
            onChange={(event) => onGoal(event.target.value)}
            placeholder="Predict churn, explain revenue drivers, find the best forecasting setup..."
            disabled={launchDisabled}
            style={{ minHeight: 96 }}
          />
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Btn
              variant="primary"
              size="lg"
              icon={loading ? Loader2 : Play}
              onClick={onStart}
              disabled={launchDisabled}
            >
              {loading ? "starting…" : "launch run"}
            </Btn>
            <span className="eyebrow" style={{ color: "var(--fg-4)" }}>
              {datasets.length} datasets · {health?.openai_configured ? "backend ready" : "backend offline"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
