import { useCallback, useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import {
  AlertTriangle,
  ArrowLeft,
  Database,
  FlaskConical,
  Loader2,
  Upload
} from "lucide-react";

import { getRunCharts, scoreRunWithDataset, scoreRunWithFile } from "../api";
import { Btn, Pill } from "../ui";
import { formatDate } from "../utils";

// ─── helpers ─────────────────────────────────────────────────────────────────

function primaryMetricLabel(run) {
  if (!run) return null;
  return run.task_type === "classification" ? "f1_weighted" : "r2";
}

function primaryMetricValue(run) {
  const key = primaryMetricLabel(run);
  if (!key) return null;
  const v = run?.best_metrics?.[key];
  return v != null ? v.toFixed(4) : null;
}

function taskBadgeTone(taskType, isTimeSeries) {
  if (isTimeSeries) return "warn";
  if (taskType === "classification") return "success";
  return "neutral";
}

function taskLabel(run) {
  if (!run) return "—";
  const base = run.task_type ?? "unknown";
  const diag = run.diagnostics ?? {};
  if (diag.is_time_series) return `${base} · time-series`;
  return base;
}

function parseFigure(figureJson) {
  try {
    return JSON.parse(figureJson);
  } catch {
    return null;
  }
}

// ─── sub-components ──────────────────────────────────────────────────────────

function MetricCard({ label, value }) {
  return (
    <div className="mr-metric-card">
      <span className="mr-metric-card__label">{label}</span>
      <span className="mr-metric-card__value">{value ?? "—"}</span>
    </div>
  );
}

function ChartPanel({ title, figureJson }) {
  const fig = parseFigure(figureJson);
  if (!fig) return null;
  return (
    <div className="mr-chart-panel">
      <div className="mr-chart-panel__title">{title}</div>
      <Plot
        data={fig.data ?? []}
        layout={{
          ...(fig.layout ?? {}),
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "var(--fg-1)", family: "JetBrains Mono, monospace", size: 11 },
          margin: { t: 40, r: 16, b: 48, l: 60 },
          xaxis: { ...(fig.layout?.xaxis ?? {}), gridcolor: "var(--ink-700)", zerolinecolor: "var(--ink-700)" },
          yaxis: { ...(fig.layout?.yaxis ?? {}), gridcolor: "var(--ink-700)", zerolinecolor: "var(--ink-700)" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", minHeight: 280 }}
        useResizeHandler
      />
    </div>
  );
}

function PredictionTable({ predictions, probabilities, taskType, featureColumns, inputDf }) {
  const rows = predictions.slice(0, 50);
  const hasCols = Array.isArray(featureColumns) && featureColumns.length > 0;
  const hasInput = Array.isArray(inputDf) && inputDf.length > 0;
  const showCols = hasCols ? featureColumns.slice(0, 5) : [];

  // Build inline prediction distribution chart
  const distLayout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "var(--fg-1)", family: "JetBrains Mono, monospace", size: 10 },
    margin: { t: 28, r: 10, b: 36, l: 48 },
    xaxis: { gridcolor: "var(--ink-700)", zerolinecolor: "var(--ink-700)" },
    yaxis: { gridcolor: "var(--ink-700)", zerolinecolor: "var(--ink-700)" },
  };

  let distData = null;
  if (taskType === "classification") {
    const counts = {};
    predictions.forEach((p) => { counts[String(p)] = (counts[String(p)] ?? 0) + 1; });
    const labels = Object.keys(counts);
    distData = [{ type: "bar", x: labels, y: labels.map((l) => counts[l]), marker: { color: "#818CF8" } }];
  } else {
    const nums = predictions.filter((p) => typeof p === "number" || !isNaN(Number(p))).map(Number);
    if (nums.length > 0) {
      distData = [{ type: "histogram", x: nums, marker: { color: "#06D7E8" }, nbinsx: 20 }];
    }
  }

  return (
    <div className="mr-pred-results">
      <div className="mr-pred-results__header">
        <span className="eyebrow">predictions — {predictions.length} rows</span>
      </div>

      {distData ? (
        <div className="mr-chart-panel" style={{ marginBottom: 16 }}>
          <div className="mr-chart-panel__title">
            {taskType === "classification" ? "Prediction Class Distribution" : "Prediction Distribution"}
          </div>
          <Plot
            data={distData}
            layout={{ ...distLayout, height: 200 }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </div>
      ) : null}

      <div className="mr-pred-table-wrap">
        <table className="mr-pred-table">
          <thead>
            <tr>
              <th>#</th>
              {hasInput && showCols.map((col) => <th key={col}>{col}</th>)}
              <th>prediction</th>
              {probabilities ? <th>confidence</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((pred, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                {hasInput && showCols.map((col) => (
                  <td key={col}>{String(inputDf?.[i]?.[col] ?? "—")}</td>
                ))}
                <td><strong>{String(pred)}</strong></td>
                {probabilities ? (
                  <td>{Math.max(...(probabilities[i] ?? [0])).toFixed(3)}</td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
        {predictions.length > 50 ? (
          <p className="mr-pred-table__overflow">
            Showing first 50 of {predictions.length} rows
          </p>
        ) : null}
      </div>
    </div>
  );
}

// ─── main component ──────────────────────────────────────────────────────────

export default function ModelReviewPage({ projectId, runs, datasets, onBack }) {
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [activeTab, setActiveTab] = useState("eval");
  const [charts, setCharts] = useState([]);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [chartsError, setChartsError] = useState(null);

  // Score tab state
  const [scoreMode, setScoreMode] = useState("dataset"); // "dataset" | "upload"
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [scoreFile, setScoreFile] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState(null);
  const [scoreResult, setScoreResult] = useState(null);
  const [inputRows, setInputRows] = useState(null);
  const fileRef = useRef(null);

  const selectedRun = runs.find((r) => r.run_id === selectedRunId) ?? null;

  // Auto-select first run
  useEffect(() => {
    if (runs.length > 0 && !selectedRunId) {
      setSelectedRunId(runs[0].run_id);
    }
  }, [runs, selectedRunId]);

  // Reset dataset selector to first dataset when project/datasets change
  useEffect(() => {
    if (datasets.length > 0 && !selectedDatasetId) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [datasets, selectedDatasetId]);

  // Load charts when run changes
  const loadCharts = useCallback(async (runId) => {
    if (!projectId || !runId) return;
    setChartsLoading(true);
    setChartsError(null);
    setCharts([]);
    try {
      const result = await getRunCharts(projectId, runId);
      setCharts(result.charts ?? []);
    } catch (err) {
      setChartsError(err instanceof Error ? err.message : String(err));
    } finally {
      setChartsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    if (selectedRunId && activeTab === "eval") {
      void loadCharts(selectedRunId);
    }
  }, [selectedRunId, activeTab, loadCharts]);

  const handleRunSelect = (runId) => {
    setSelectedRunId(runId);
    setScoreResult(null);
    setScoreError(null);
    setInputRows(null);
  };

  const handleScore = async () => {
    if (!projectId || !selectedRunId) return;
    setScoring(true);
    setScoreError(null);
    setScoreResult(null);
    setInputRows(null);
    try {
      let result;
      if (scoreMode === "upload" && scoreFile) {
        // Read file to show input rows in the table
        const text = await scoreFile.text();
        const lines = text.trim().split("\n");
        if (lines.length > 1) {
          const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
          const dataRows = lines.slice(1, 52).map((line) => {
            const vals = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
            const row = {};
            headers.forEach((h, i) => { row[h] = vals[i] ?? ""; });
            return row;
          });
          setInputRows(dataRows);
        }
        result = await scoreRunWithFile(projectId, selectedRunId, scoreFile);
      } else {
        result = await scoreRunWithDataset(projectId, selectedRunId, selectedDatasetId);
      }
      setScoreResult(result);
    } catch (err) {
      setScoreError(err instanceof Error ? err.message : String(err));
    } finally {
      setScoring(false);
    }
  };

  const metricEntries = selectedRun?.best_metrics
    ? Object.entries(selectedRun.best_metrics).slice(0, 8)
    : [];

  return (
    <div className="mr-page">
      {/* Run list (left panel) */}
      <aside className="mr-sidebar">
        <div className="mr-sidebar__header">
          <button type="button" className="mr-back-btn" onClick={onBack}>
            <ArrowLeft size={13} strokeWidth={1.75} />
            back
          </button>
          <span className="eyebrow" style={{ marginTop: 12 }}>
            trained models
            <span style={{ color: "var(--fg-3)", marginLeft: 6 }}>{runs.length}</span>
          </span>
        </div>

        <div className="mr-run-list">
          {runs.length === 0 ? (
            <p className="empty-note">No trained models found.<br />Run Autopilot to train models.</p>
          ) : null}
          {runs.map((run) => {
            const metric = primaryMetricValue(run);
            const metricKey = primaryMetricLabel(run);
            const isTs = run.diagnostics?.is_time_series ?? false;
            return (
              <button
                key={run.run_id}
                type="button"
                className={`mr-run-row${selectedRunId === run.run_id ? " is-selected" : ""}`}
                onClick={() => handleRunSelect(run.run_id)}
              >
                <div className="mr-run-row__top">
                  <span className="mr-run-row__model">{run.best_model_name ?? "Model"}</span>
                  <Pill tone={taskBadgeTone(run.task_type, isTs)} style={{ fontSize: 9 }}>
                    {isTs ? "TS" : run.task_type === "classification" ? "CLF" : "REG"}
                  </Pill>
                </div>
                <div className="mr-run-row__meta">
                  <span>{run.target_column ?? "—"}</span>
                  {metric ? (
                    <span style={{ color: "var(--cyan-500)" }}>{metricKey}: {metric}</span>
                  ) : null}
                </div>
                <div className="mr-run-row__date">{formatDate(run.trained_at)}</div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* Main content */}
      <div className="mr-main">
        {!selectedRun ? (
          <div className="mr-empty">
            <FlaskConical size={40} strokeWidth={1} style={{ color: "var(--fg-4)" }} />
            <p>Select a model from the list to review it.</p>
          </div>
        ) : (
          <>
            {/* Run info header */}
            <div className="mr-run-header">
              <div className="mr-run-header__left">
                <h2 className="mr-run-header__title">{selectedRun.best_model_name ?? "Model"}</h2>
                <Pill tone={taskBadgeTone(selectedRun.task_type, selectedRun.diagnostics?.is_time_series)}>
                  {taskLabel(selectedRun)}
                </Pill>
              </div>
              <div className="mr-run-header__meta">
                <span>target: <strong>{selectedRun.target_column ?? "—"}</strong></span>
                <span style={{ color: "var(--fg-3)" }}>·</span>
                <span>{selectedRun.row_count?.toLocaleString() ?? "?"} rows</span>
                <span style={{ color: "var(--fg-3)" }}>·</span>
                <span>{formatDate(selectedRun.trained_at)}</span>
              </div>
            </div>

            {/* Metrics grid */}
            {metricEntries.length > 0 ? (
              <div className="mr-metrics-grid">
                {metricEntries.map(([key, val]) => (
                  <MetricCard key={key} label={key} value={typeof val === "number" ? val.toFixed(4) : String(val)} />
                ))}
              </div>
            ) : null}

            {/* Feature columns hint */}
            {selectedRun.feature_columns?.length > 0 ? (
              <div className="mr-features-hint">
                <span className="eyebrow">features ({selectedRun.feature_columns.length})</span>
                <span className="mr-features-hint__list">
                  {selectedRun.feature_columns.slice(0, 10).join(", ")}
                  {selectedRun.feature_columns.length > 10 ? ` …+${selectedRun.feature_columns.length - 10} more` : ""}
                </span>
              </div>
            ) : null}

            {/* Tabs */}
            <div className="mr-tabs">
              <button
                type="button"
                className={`mr-tab${activeTab === "eval" ? " is-active" : ""}`}
                onClick={() => setActiveTab("eval")}
              >
                Test Set Evaluation
              </button>
              <button
                type="button"
                className={`mr-tab${activeTab === "score" ? " is-active" : ""}`}
                onClick={() => setActiveTab("score")}
              >
                Score New Data
              </button>
            </div>

            {/* Tab content */}
            {activeTab === "eval" ? (
              <div className="mr-tab-content">
                {chartsLoading ? (
                  <div className="mr-loading">
                    <Loader2 size={20} strokeWidth={1.75} className="spin" />
                    <span>Building charts…</span>
                  </div>
                ) : chartsError ? (
                  <div className="mr-error">
                    <AlertTriangle size={14} strokeWidth={1.75} />
                    <span>{chartsError}</span>
                    <Btn variant="ghost" size="sm" onClick={() => void loadCharts(selectedRunId)}>
                      retry
                    </Btn>
                  </div>
                ) : charts.length === 0 ? (
                  <div className="mr-empty mr-empty--inline">
                    <p>No diagnostic charts available for this run.<br />Charts require stored test data from the Autopilot session.</p>
                  </div>
                ) : (
                  <div className="mr-charts-grid">
                    {charts.map((chart) => (
                      <ChartPanel key={chart.title} title={chart.title} figureJson={chart.figure_json} />
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="mr-tab-content">
                <div className="mr-score-form">
                  <span className="eyebrow">data source</span>
                  <div className="mr-score-mode-toggle">
                    <button
                      type="button"
                      className={`mr-mode-btn${scoreMode === "dataset" ? " is-active" : ""}`}
                      onClick={() => setScoreMode("dataset")}
                    >
                      <Database size={12} strokeWidth={1.75} />
                      existing dataset
                    </button>
                    <button
                      type="button"
                      className={`mr-mode-btn${scoreMode === "upload" ? " is-active" : ""}`}
                      onClick={() => setScoreMode("upload")}
                    >
                      <Upload size={12} strokeWidth={1.75} />
                      upload CSV
                    </button>
                  </div>

                  {scoreMode === "dataset" ? (
                    <select
                      className="field"
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                      disabled={datasets.length === 0}
                    >
                      {datasets.length === 0 ? (
                        <option>No datasets available</option>
                      ) : null}
                      {datasets.map((ds) => (
                        <option key={ds.id} value={ds.id}>
                          {ds.name} ({ds.row_count?.toLocaleString() ?? "?"} rows)
                        </option>
                      ))}
                    </select>
                  ) : (
                    <label
                      className="field mr-file-label"
                      style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: 8, color: "var(--fg-2)" }}
                    >
                      <Upload size={13} strokeWidth={1.75} />
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {scoreFile?.name ?? "Choose CSV file"}
                      </span>
                      <input
                        key={fileInputKey}
                        ref={fileRef}
                        type="file"
                        accept=".csv"
                        style={{ display: "none" }}
                        onChange={(e) => {
                          setScoreFile(e.target.files?.[0] ?? null);
                          setScoreResult(null);
                        }}
                      />
                    </label>
                  )}

                  {selectedRun.feature_columns?.length > 0 ? (
                    <p className="mr-score-hint">
                      Expected columns: {selectedRun.feature_columns.slice(0, 6).join(", ")}
                      {selectedRun.feature_columns.length > 6 ? " …" : ""}
                    </p>
                  ) : null}

                  <Btn
                    variant="primary"
                    size="sm"
                    icon={scoring ? Loader2 : FlaskConical}
                    disabled={scoring || (scoreMode === "upload" && !scoreFile) || (scoreMode === "dataset" && !selectedDatasetId)}
                    onClick={() => void handleScore()}
                  >
                    {scoring ? "scoring…" : "run predictions"}
                  </Btn>
                </div>

                {scoreError ? (
                  <div className="mr-error" style={{ marginTop: 12 }}>
                    <AlertTriangle size={14} strokeWidth={1.75} />
                    <span>{scoreError}</span>
                  </div>
                ) : null}

                {scoreResult ? (
                  <PredictionTable
                    predictions={scoreResult.predictions}
                    probabilities={scoreResult.probabilities}
                    taskType={scoreResult.task_type}
                    featureColumns={selectedRun.feature_columns}
                    inputDf={inputRows}
                  />
                ) : null}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
