import { Loader2, Play } from "lucide-react";

import { Btn } from "../ui";

export default function EmptyCanvas({ project, datasets, health, goal, onGoal, onStart, loading, disabledLaunch }) {
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
          <textarea
            className="field"
            value={goal}
            onChange={(event) => onGoal(event.target.value)}
            placeholder="Predict churn, explain revenue drivers, find the best forecasting setup..."
            disabled={disabledLaunch}
            style={{ minHeight: 96 }}
          />
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Btn
              variant="primary"
              size="lg"
              icon={loading ? Loader2 : Play}
              onClick={onStart}
              disabled={disabledLaunch}
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
