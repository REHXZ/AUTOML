import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Circle,
  ClipboardList,
  Database,
  Download,
  FileText,
  FolderPlus,
  Loader2,
  MessageSquare,
  Moon,
  Play,
  Radio,
  RefreshCw,
  Send,
  Sun,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import ReactMarkdown from "react-markdown";

import {
  connectSessionEvents,
  createProject,
  deleteSession,
  getHealth,
  getSession,
  isTerminalStatus,
  listDatasets,
  listProjects,
  listSessions,
  notebookUrl,
  sendFollowUp,
  startSession,
  submitAnswers,
  uploadDataset
} from "./api";
import { AGENTS } from "./constants";
import {
  formatDate,
  getAgentState,
  getPhaseState,
  maxStepIndex,
  mergeStep,
  normalizeQuestions,
  parseFigure,
  shortDetail,
  statusText,
  stepKindLabel,
  visibleActivitySteps
} from "./utils";

const THEME_STORAGE_KEY = "aiml-dashboard-theme";

function getInitialTheme() {
  if (typeof window === "undefined") return "light";
  let stored = null;
  try {
    stored = window.localStorage?.getItem(THEME_STORAGE_KEY) ?? null;
  } catch {
    stored = null;
  }
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)")?.matches ? "dark" : "light";
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [projects, setProjects] = useState([]);
  const [projectId, setProjectId] = useState("");
  const [datasets, setDatasets] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [goal, setGoal] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [datasetFile, setDatasetFile] = useState(null);
  const [datasetName, setDatasetName] = useState("");
  const [datasetTableName, setDatasetTableName] = useState("");
  const [fileInputKey, setFileInputKey] = useState(0);
  const [notice, setNotice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [error, setError] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [selectedAgentIds, setSelectedAgentIds] = useState([]);
  const [theme, setTheme] = useState(getInitialTheme);
  const closeStreamRef = useRef(null);

  const selectedProject = projects.find((project) => project.id === projectId) ?? null;
  const questions = useMemo(
    () => normalizeQuestions(activeSession?.pending_step ?? null),
    [activeSession?.pending_step]
  );
  const [answers, setAnswers] = useState([]);

  const refreshSessions = useCallback(async (id) => {
    const records = await listSessions(id);
    setSessions(records);
  }, []);

  const closeStream = useCallback(() => {
    closeStreamRef.current?.();
    closeStreamRef.current = null;
    setStreaming(false);
  }, []);

  const refreshSession = useCallback(async (id, sessionId) => {
    const loaded = await getSession(id, sessionId);
    setActiveSession(loaded);
    return loaded;
  }, []);

  const openEventStream = useCallback(
    (id, sessionId, fromIndex) => {
      closeStream();
      setStreaming(true);
      closeStreamRef.current = connectSessionEvents(id, sessionId, fromIndex, {
        onStep: (step) => {
          setActiveSession((current) => {
            if (!current || current.session_id !== sessionId) return current;
            return { ...current, steps: mergeStep(current.steps, step) };
          });
        },
        onStatus: async (payload) => {
          setStreaming(false);
          setActiveSession((current) => {
            if (!current || current.session_id !== sessionId) return current;
            return {
              ...current,
              status: payload.status,
              pending_step: payload.pending_step ?? current.pending_step,
              error: payload.error ?? current.error
            };
          });
          if (isTerminalStatus(payload.status)) {
            await Promise.all([refreshSessions(id), refreshSession(id, sessionId)]);
          }
        },
        onHeartbeat: (payload) => {
          setActiveSession((current) => {
            if (!current || current.session_id !== sessionId) return current;
            return { ...current, status: payload.status };
          });
        },
        onError: () => {
          setStreaming(false);
          setError("Live event stream disconnected. Refresh the session to reconnect.");
        }
      });
    },
    [closeStream, refreshSession, refreshSessions]
  );

  const loadProjectData = useCallback(async (id) => {
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const [projectDatasets, projectSessions] = await Promise.all([
        listDatasets(id),
        listSessions(id)
      ]);
      setDatasets(projectDatasets);
      setSessions(projectSessions);
      setActiveSession(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const bootstrap = useCallback(async () => {
    setLoading(true);
    setError(null);
    setNotice(null);
    closeStream();
    try {
      const [nextHealth, nextProjects] = await Promise.all([getHealth(), listProjects()]);
      setHealth(nextHealth);
      setProjects(nextProjects);
      const nextProjectId = projectId || nextProjects[0]?.id || "";
      setProjectId(nextProjectId);
      if (nextProjectId) await loadProjectData(nextProjectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [closeStream, loadProjectData, projectId]);

  useEffect(() => {
    void bootstrap();
    return () => closeStream();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    try {
      window.localStorage?.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Theme still works when browser storage is unavailable.
    }
  }, [theme]);

  useEffect(() => {
    setAnswers(questions.map((question) => question.recommendation ?? ""));
  }, [questions]);

  useEffect(() => {
    setSelectedAgentIds([]);
  }, [projectId, activeSession?.session_id]);

  const handleProjectChange = async (id) => {
    closeStream();
    setProjectId(id);
    setDatasetFile(null);
    setDatasetName("");
    setDatasetTableName("");
    setFileInputKey((key) => key + 1);
    await loadProjectData(id);
  };

  const handleCreateProject = async () => {
    if (!projectName.trim()) return;
    closeStream();
    setLoading(true);
    setCreatingProject(true);
    setError(null);
    setNotice(null);
    try {
      const project = await createProject(projectName.trim(), projectDescription.trim());
      const nextProjects = await listProjects();
      setProjects(nextProjects);
      setProjectId(project.id);
      setProjectName("");
      setProjectDescription("");
      setDatasetFile(null);
      setDatasetName("");
      setDatasetTableName("");
      setFileInputKey((key) => key + 1);
      await loadProjectData(project.id);
      setNotice(`Created project "${project.name}".`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingProject(false);
      setLoading(false);
    }
  };

  const handleUploadDataset = async () => {
    if (!projectId || !datasetFile) return;
    setLoading(true);
    setUploadingDataset(true);
    setError(null);
    setNotice(null);
    try {
      await uploadDataset(projectId, {
        file: datasetFile,
        name: datasetName,
        tableName: datasetTableName
      });
      const [nextDatasets, nextProjects] = await Promise.all([
        listDatasets(projectId),
        listProjects()
      ]);
      setDatasets(nextDatasets);
      setProjects(nextProjects);
      setDatasetFile(null);
      setDatasetName("");
      setDatasetTableName("");
      setFileInputKey((key) => key + 1);
      setNotice(`Uploaded "${datasetFile.name}".`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploadingDataset(false);
      setLoading(false);
    }
  };

  const handleOpenSession = async (record) => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const loaded = await refreshSession(projectId, record.session_id);
      if (loaded.status === "running") {
        openEventStream(projectId, loaded.session_id, maxStepIndex(loaded.steps));
      } else {
        closeStream();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleStart = async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const job = await startSession(projectId, goal.trim());
      const loaded = await refreshSession(projectId, job.session_id);
      await refreshSessions(projectId);
      openEventStream(projectId, loaded.session_id, maxStepIndex(loaded.steps));
      setGoal("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (record) => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      await deleteSession(projectId, record.session_id);
      if (activeSession?.session_id === record.session_id) {
        closeStream();
        setActiveSession(null);
      }
      await refreshSessions(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitAnswers = async () => {
    if (!projectId || !activeSession) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      await submitAnswers(projectId, activeSession.session_id, answers.map((answer) => answer.trim()));
      const loaded = await refreshSession(projectId, activeSession.session_id);
      openEventStream(projectId, activeSession.session_id, maxStepIndex(loaded.steps));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleFollowUp = async () => {
    if (!projectId || !activeSession || !followUp.trim()) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      await sendFollowUp(projectId, activeSession.session_id, followUp.trim());
      const loaded = await refreshSession(projectId, activeSession.session_id);
      openEventStream(projectId, activeSession.session_id, maxStepIndex(loaded.steps));
      setFollowUp("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const stats = getStats(activeSession);
  const agentStates = useMemo(() => getAgentState(activeSession), [activeSession]);
  const visibleTimelineSteps = useMemo(() => {
    const steps = visibleActivitySteps(activeSession?.steps ?? []);
    if (selectedAgentIds.length === 0) return steps;
    return steps.filter((step) => selectedAgentIds.includes(step.agent));
  }, [activeSession?.steps, selectedAgentIds]);
  const selectedAgents = useMemo(
    () => agentStates.filter((agent) => selectedAgentIds.includes(agent.id)),
    [agentStates, selectedAgentIds]
  );
  const disabledLaunch = loading || !projectId || datasets.length === 0;

  const toggleAgentFilter = (agentId) => {
    setSelectedAgentIds((current) =>
      current.includes(agentId)
        ? current.filter((id) => id !== agentId)
        : [...current, agentId]
    );
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            <Activity size={20} />
          </div>
          <div>
            <h1>AIML Autopilot</h1>
            <p>Agent operations</p>
          </div>
        </div>

        <label className="field-label" htmlFor="project-select">
          Project
        </label>
        <select
          id="project-select"
          value={projectId}
          onChange={(event) => void handleProjectChange(event.target.value)}
          disabled={loading || projects.length === 0}
        >
          {projects.length === 0 ? <option>No projects found</option> : null}
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>

        <form
          className="project-create"
          onSubmit={(event) => {
            event.preventDefault();
            void handleCreateProject();
          }}
        >
          <div className="side-header">
            <span>New Project</span>
          </div>
          <input
            value={projectName}
            onChange={(event) => setProjectName(event.target.value)}
            placeholder="Project name"
            disabled={loading}
          />
          <textarea
            className="compact-textarea"
            value={projectDescription}
            onChange={(event) => setProjectDescription(event.target.value)}
            placeholder="Description"
            disabled={loading}
          />
          <button className="secondary-button sidebar-button" disabled={loading || !projectName.trim()}>
            {creatingProject ? <Loader2 className="spin" size={16} /> : <FolderPlus size={16} />}
            Create
          </button>
        </form>

        <div className="side-header">
          <span>Sessions</span>
          <button className="icon-button" onClick={() => projectId && void refreshSessions(projectId)}>
            <RefreshCw size={16} />
          </button>
        </div>
        <div className="session-list">
          {sessions.map((record) => (
            <div
              className={`session-row ${activeSession?.session_id === record.session_id ? "selected" : ""}`}
              key={record.session_id}
            >
              <button className="session-open" onClick={() => void handleOpenSession(record)}>
                <span className={`status-dot ${record.status}`} />
                <span>
                  <strong>{record.title || record.session_id}</strong>
                  <small>
                    {record.step_count} steps - {formatDate(record.updated_at)}
                  </small>
                </span>
              </button>
              <button
                className="icon-button danger"
                onClick={() => void handleDelete(record)}
                disabled={record.status === "running"}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
          {sessions.length === 0 ? <p className="empty-note">No saved sessions</p> : null}
        </div>
      </aside>

      <main className="dashboard">
        <header className="topbar">
          <div>
            <p className="eyebrow">Current Workspace</p>
            <h2>{selectedProject?.name ?? "No project selected"}</h2>
            <p className="muted">{selectedProject?.description || selectedProject?.id || "Create a project in the existing app first."}</p>
          </div>
          <div className="top-actions">
            <HealthPill health={health} />
            <ThemeToggle theme={theme} onToggle={() => setTheme((value) => (value === "dark" ? "light" : "dark"))} />
            <button className="secondary-button" onClick={() => void bootstrap()}>
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        </header>

        {error ? (
          <div className="alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        {notice ? (
          <div className="notice">
            <CheckCircle2 size={18} />
            <span>{notice}</span>
          </div>
        ) : null}

        <section className="launch-band">
          <div className="launch-copy">
            <p className="eyebrow">Launch</p>
            <h3>Autopilot run</h3>
          </div>
          <textarea
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            placeholder="Predict churn, explain revenue drivers, find the best forecasting setup..."
            disabled={disabledLaunch}
          />
          <button className="primary-button" onClick={() => void handleStart()} disabled={disabledLaunch}>
            {loading ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
            Start
          </button>
        </section>

        <section className="metric-grid">
          <Metric icon={Radio} label="Status" value={statusText(activeSession?.status)} detail={streaming ? "live stream connected" : "stream idle"} />
          <Metric icon={ClipboardList} label="Steps" value={String(stats.steps)} detail={`${stats.charts} charts captured`} />
          <Metric icon={Database} label="Datasets" value={String(datasets.length)} detail={`${stats.generatedDatasets} generated`} />
          <Metric icon={FileText} label="Training Runs" value={String(stats.trainingRuns)} detail={`${stats.notes} notebook notes`} />
        </section>

        <div className="work-grid">
          <section className="panel">
            <PanelTitle icon={Database} title="Datasets" />
            <DatasetUpload
              file={datasetFile}
              fileInputKey={fileInputKey}
              name={datasetName}
              tableName={datasetTableName}
              loading={uploadingDataset}
              disabled={loading || !projectId}
              onFile={setDatasetFile}
              onName={setDatasetName}
              onTableName={setDatasetTableName}
              onSubmit={() => void handleUploadDataset()}
            />
            <DatasetTable datasets={datasets} />
          </section>

          <section className="panel">
            <PanelTitle icon={Activity} title="Lifecycle" />
            <PhaseRail session={activeSession} />
          </section>
        </div>

        <section className="agent-lanes">
          <PanelTitle icon={Radio} title="Agents" />
          <div className="agent-grid">
            {agentStates.map((agent) => {
              const Icon = agent.icon;
              const selected = selectedAgentIds.includes(agent.id);
              return (
                <button
                  type="button"
                  className={`agent-card ${agent.tone} ${selected ? "selected" : ""}`}
                  aria-pressed={selected}
                  key={agent.id}
                  onClick={() => toggleAgentFilter(agent.id)}
                  title={`${selected ? "Remove" : "Show"} ${agent.title} activity`}
                >
                  <div className="agent-card-head">
                    <Icon size={18} />
                    <strong>{agent.title}</strong>
                    <span className={`agent-state ${agent.state}`}>{agent.state}</span>
                  </div>
                  <p>{shortDetail(agent.lastStep)}</p>
                  <small>{agent.steps} steps - {agent.charts} charts</small>
                </button>
              );
            })}
          </div>
        </section>

        {activeSession?.status === "waiting_for_input" ? (
          <QuestionPanel
            questions={questions}
            answers={answers}
            onAnswer={setAnswers}
            onSubmit={() => void handleSubmitAnswers()}
            loading={loading}
          />
        ) : null}

        <div className="content-grid">
          <section className="panel timeline-panel">
            <div className="timeline-toolbar">
              <PanelTitle icon={ClipboardList} title="Activity Timeline" />
              {selectedAgents.length > 0 ? (
                <div className="filter-chips" aria-label="Selected agent filters">
                  {selectedAgents.map((agent) => (
                    <button
                      type="button"
                      className="filter-chip"
                      key={agent.id}
                      onClick={() => toggleAgentFilter(agent.id)}
                      title={`Remove ${agent.title} filter`}
                    >
                      {agent.title}
                    </button>
                  ))}
                  <button
                    type="button"
                    className="icon-button filter-clear"
                    onClick={() => setSelectedAgentIds([])}
                    title="Show all agents"
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <span className="filter-status">All agents</span>
              )}
            </div>
            <Timeline steps={visibleTimelineSteps} filtered={selectedAgentIds.length > 0} theme={theme} />
          </section>

          <aside className="results-stack">
            <ResultsPanel session={activeSession} projectId={projectId} />
            <FollowUpPanel
              session={activeSession}
              value={followUp}
              onChange={setFollowUp}
              onSend={() => void handleFollowUp()}
              loading={loading}
            />
          </aside>
        </div>
      </main>
    </div>
  );
}

function HealthPill({ health }) {
  if (!health) {
    return (
      <span className="health-pill error">
        <AlertTriangle size={15} />
        API offline
      </span>
    );
  }
  return (
    <span className={`health-pill ${health.openai_configured ? "ok" : "warn"}`}>
      {health.openai_configured ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
      {health.openai_configured ? "Backend ready" : "Key missing"}
    </span>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  detail
}) {
  return (
    <div className="metric">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function ThemeToggle({ theme, onToggle }) {
  const dark = theme === "dark";
  const Icon = dark ? Sun : Moon;
  return (
    <button
      type="button"
      className="secondary-button theme-toggle"
      onClick={onToggle}
      title={dark ? "Switch to light theme" : "Switch to dark theme"}
    >
      <Icon size={16} />
      <span>{dark ? "Light" : "Dark"}</span>
    </button>
  );
}

function PanelTitle({ icon: Icon, title }) {
  return (
    <div className="panel-title">
      <Icon size={17} />
      <h3>{title}</h3>
    </div>
  );
}

function DatasetUpload({
  file,
  fileInputKey,
  name,
  tableName,
  loading,
  disabled,
  onFile,
  onName,
  onTableName,
  onSubmit
}) {
  return (
    <form
      className="upload-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <label className={`file-picker ${disabled ? "disabled" : ""}`}>
        <Upload size={16} />
        <span>{file?.name ?? "Choose file"}</span>
        <input
          key={fileInputKey}
          type="file"
          accept=".csv,.xlsx,.xls,.json,.db,.sqlite,.sqlite3"
          disabled={disabled}
          onChange={(event) => onFile(event.target.files?.[0] ?? null)}
        />
      </label>
      <input
        value={name}
        onChange={(event) => onName(event.target.value)}
        placeholder="Dataset name"
        disabled={disabled}
      />
      <input
        className="table-name-input"
        value={tableName}
        onChange={(event) => onTableName(event.target.value)}
        placeholder="SQLite table"
        disabled={disabled}
      />
      <button className="secondary-button" disabled={disabled || loading || !file}>
        {loading ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
        Upload
      </button>
    </form>
  );
}

function DatasetTable({ datasets }) {
  if (datasets.length === 0) {
    return <p className="empty-note">No datasets registered for this project.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Source</th>
            <th>Rows</th>
            <th>Columns</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((dataset) => (
            <tr key={dataset.id}>
              <td>{dataset.name}</td>
              <td>{dataset.source_type}</td>
              <td>{dataset.row_count.toLocaleString()}</td>
              <td>{dataset.column_count.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PhaseRail({ session }) {
  return (
    <div className="phase-rail">
      {getPhaseState(session).map((phase) => (
        <div className={`phase-step ${phase.state}`} key={phase.id}>
          <Circle size={12} />
          <span>{phase.title}</span>
          <small>{phase.count}</small>
        </div>
      ))}
    </div>
  );
}

function QuestionPanel({
  questions,
  answers,
  onAnswer,
  onSubmit,
  loading
}) {
  return (
    <section className="question-panel">
      <PanelTitle icon={MessageSquare} title="Input Needed" />
      {questions.map((question, index) => (
        <div className="question-item" key={`${question.question}-${index}`}>
          <strong>{question.question}</strong>
          {question.explanation ? <p>{question.explanation}</p> : null}
          {question.alternatives?.length ? (
            <div className="alternatives">
              {question.alternatives.map((alternative) => (
                <button
                  key={alternative}
                  type="button"
                  onClick={() => {
                    const next = [...answers];
                    next[index] = alternative;
                    onAnswer(next);
                  }}
                >
                  {alternative}
                </button>
              ))}
            </div>
          ) : null}
          <textarea
            value={answers[index] ?? ""}
            onChange={(event) => {
              const next = [...answers];
              next[index] = event.target.value;
              onAnswer(next);
            }}
          />
        </div>
      ))}
      <button className="primary-button" onClick={onSubmit} disabled={loading}>
        {loading ? <Loader2 className="spin" size={17} /> : <Send size={17} />}
        Submit Answers
      </button>
    </section>
  );
}

function Timeline({ steps, filtered, theme }) {
  if (steps.length === 0) {
    return (
      <p className="empty-note">
        {filtered ? "No activity for selected agents." : "Open or start a session to see activity."}
      </p>
    );
  }
  return (
    <div className="timeline">
      {[...steps].reverse().map((step) => (
        <StepCard key={step.index} step={step} theme={theme} />
      ))}
    </div>
  );
}

function StepCard({ step, theme }) {
  const agent = AGENTS.find((item) => item.id === step.agent);
  const figure = parseFigure(step);
  const figureLayout = figure?.layout ?? {};
  const chartLayout = {
    ...figureLayout,
    autosize: true,
    height: 360,
    margin: { t: 36, r: 18, b: 48, l: 52, ...(figureLayout.margin ?? {}) },
    ...(theme === "dark"
      ? {
          paper_bgcolor: figureLayout.paper_bgcolor ?? "#141e2b",
          plot_bgcolor: figureLayout.plot_bgcolor ?? "#141e2b",
          font: {
            ...(figureLayout.font ?? {}),
            color: figureLayout.font?.color ?? "#e6eef8"
          }
        }
      : {})
  };
  return (
    <article className={`step-card ${step.kind}`}>
      <div className="step-meta">
        <span>#{step.index}</span>
        <span>{agent?.title ?? step.agent}</span>
        <span>{stepKindLabel(step.kind)}</span>
      </div>
      <h4>{step.title}</h4>
      {step.kind === "chart" && figure ? (
        <Plot
          data={figure.data ?? []}
          layout={chartLayout}
          config={{ displaylogo: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%", height: "360px" }}
        />
      ) : step.detail ? (
        <div className="markdown">
          <ReactMarkdown>{step.detail}</ReactMarkdown>
        </div>
      ) : null}
    </article>
  );
}

function ResultsPanel({ session, projectId }) {
  if (!session) {
    return (
      <section className="panel">
        <PanelTitle icon={FileText} title="Results" />
        <p className="empty-note">Open or start a session.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <PanelTitle icon={FileText} title="Results" />
      <div className="result-actions">
        <a className="secondary-button" href={notebookUrl(projectId, session.session_id)}>
          <Download size={16} />
          Notebook
        </a>
      </div>

      {session.training_runs.length ? (
        <>
          <h4>Training Runs</h4>
          <div className="table-wrap compact">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Model</th>
                  <th>Target</th>
                  <th>Metrics</th>
                </tr>
              </thead>
              <tbody>
                {session.training_runs.map((run, index) => (
                  <tr key={`${run.run_id ?? "run"}-${index}`}>
                    <td>{run.run_id ?? index + 1}</td>
                    <td>{run.best_model ?? "-"}</td>
                    <td>{run.target ?? "-"}</td>
                    <td>{formatMetrics(run.best_metrics)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      {session.new_datasets.length ? (
        <>
          <h4>Generated Datasets</h4>
          <ul className="plain-list">
            {session.new_datasets.map((dataset) => (
              <li key={dataset.id}>{dataset.name} - {dataset.row_count.toLocaleString()} rows</li>
            ))}
          </ul>
        </>
      ) : null}

      {session.notebook.length ? (
        <>
          <h4>Notebook Notes</h4>
          <ul className="plain-list">
            {session.notebook.slice(-6).map((note, index) => (
              <li key={`${note}-${index}`}>{note}</li>
            ))}
          </ul>
        </>
      ) : null}

      {session.strategy_summary ? (
        <>
          <h4>Final Report</h4>
          <div className="markdown report">
            <ReactMarkdown>{session.strategy_summary}</ReactMarkdown>
          </div>
        </>
      ) : null}
    </section>
  );
}

function FollowUpPanel({
  session,
  value,
  onChange,
  onSend,
  loading
}) {
  const disabled = !session || session.status === "running" || session.status === "waiting_for_input" || loading;
  return (
    <section className="panel">
      <PanelTitle icon={MessageSquare} title="Follow Up" />
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Try another tuning round focused on recall..."
        disabled={disabled}
      />
      <button className="primary-button" onClick={onSend} disabled={disabled || !value.trim()}>
        {loading ? <Loader2 className="spin" size={17} /> : <Send size={17} />}
        Send
      </button>
    </section>
  );
}

function formatMetrics(metrics) {
  if (!metrics) return "-";
  return Object.entries(metrics)
    .map(([key, value]) => `${key}: ${Number(value).toFixed(4)}`)
    .join(", ");
}

function getStats(session) {
  const steps = visibleActivitySteps(session?.steps ?? []);
  return {
    steps: steps.length,
    charts: steps.filter((step) => step.kind === "chart").length,
    trainingRuns: session?.training_runs.length ?? 0,
    generatedDatasets: session?.new_datasets.length ?? 0,
    notes: session?.notebook.length ?? 0
  };
}
