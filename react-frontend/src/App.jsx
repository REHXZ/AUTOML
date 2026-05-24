import {
  AlertTriangle,
  ChevronRight,
  Download,
  FolderPlus,
  ListOrdered,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  Search,
  Send,
  Settings2,
  SkipBack,
  SkipForward,
  Trash2,
  Upload,
  Workflow,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import DetailDrawer from "./DetailDrawer";
import LinearTimeline from "./LinearTimeline";
import SwimlaneGraph from "./SwimlaneGraph";
import {
  connectSessionEvents,
  createProject,
  deleteSession,
  getHealth,
  getSession,
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
import { Btn, IconBtn, Pill } from "./primitives";
import {
  agentFor,
  formatDate,
  getSessionStats,
  isTerminalStatus,
  maxStepIndex,
  mergeStep,
  normalizeQuestions,
  statusText,
  visibleActivitySteps
} from "./utils";

const TWEAK_KEY = "aiml-autopilot-tweaks";
const DEFAULT_TWEAKS = {
  view: "graph",
  density: "comfortable",
  showLegend: true,
  showStream: true,
  showWorkspace: true,
  replaySpeed: "live"
};

function loadTweaks() {
  if (typeof window === "undefined") return DEFAULT_TWEAKS;
  try {
    const raw = window.localStorage?.getItem(TWEAK_KEY);
    if (!raw) return DEFAULT_TWEAKS;
    return { ...DEFAULT_TWEAKS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_TWEAKS;
  }
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
  const [tweaks, setTweaks] = useState(loadTweaks);
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [hoverIndex, setHoverIndex] = useState(null);
  const [playing, setPlaying] = useState(true);
  const closeStreamRef = useRef(null);

  const selectedProject = projects.find((project) => project.id === projectId) ?? null;
  const questions = useMemo(
    () => normalizeQuestions(activeSession?.pending_step ?? null),
    [activeSession?.pending_step]
  );
  const [answers, setAnswers] = useState([]);

  const visibleSteps = useMemo(
    () => visibleActivitySteps(activeSession?.steps ?? []),
    [activeSession?.steps]
  );

  const selectedStep = useMemo(
    () =>
      selectedIndex == null
        ? null
        : visibleSteps.find((step) => step.index === selectedIndex) ?? null,
    [selectedIndex, visibleSteps]
  );

  const setTweak = useCallback((key, value) => {
    setTweaks((prev) => {
      const next = { ...prev, [key]: value };
      try {
        window.localStorage?.setItem(TWEAK_KEY, JSON.stringify(next));
      } catch {
        // storage optional
      }
      return next;
    });
  }, []);

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
      setSelectedIndex(null);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setAnswers(questions.map((question) => question.recommendation ?? ""));
  }, [questions]);

  // Keyboard: G / T to switch view, Esc handled in drawer.
  useEffect(() => {
    const onKey = (event) => {
      const tag = event.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (event.key === "g" || event.key === "G") setTweak("view", "graph");
      if (event.key === "t" || event.key === "T") setTweak("view", "timeline");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setTweak]);

  // When a new session loads, select the running step if any.
  useEffect(() => {
    if (!activeSession) {
      setSelectedIndex(null);
      return;
    }
    const visible = visibleActivitySteps(activeSession.steps ?? []);
    if (activeSession.status === "running") {
      const last = visible.at(-1);
      if (last) setSelectedIndex(last.index);
    }
  }, [activeSession?.session_id, activeSession?.status]);

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

  const handleDeleteSession = async (record) => {
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
      await submitAnswers(
        projectId,
        activeSession.session_id,
        answers.map((answer) => answer.trim())
      );
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

  const stats = useMemo(() => getSessionStats(activeSession), [activeSession]);
  const datasetSummary = useMemo(() => {
    if (!datasets.length) return "no dataset";
    const first = datasets[0];
    const rows = first.row_count?.toLocaleString?.() ?? first.row_count ?? "";
    return `${first.name} · ${rows} × ${first.column_count ?? "?"}`;
  }, [datasets]);

  const disabledLaunch = loading || !projectId || datasets.length === 0;

  return (
    <div className="app-root">
      <AppHeader
        project={selectedProject}
        session={activeSession}
        streaming={streaming}
        loading={loading}
        onRefresh={() => void bootstrap()}
        onTweaks={() => setTweaksOpen((open) => !open)}
        notebookHref={
          activeSession ? notebookUrl(projectId, activeSession.session_id) : null
        }
      />

      {error ? (
        <div className="alert-bar alert-bar--error">
          <AlertTriangle size={15} strokeWidth={1.75} />
          <span>{error}</span>
          <button type="button" className="alert-bar__close" onClick={() => setError(null)}>
            <X size={14} />
          </button>
        </div>
      ) : null}
      {notice ? (
        <div className="alert-bar alert-bar--notice">
          <span>{notice}</span>
          <button type="button" className="alert-bar__close" onClick={() => setNotice(null)}>
            <X size={14} />
          </button>
        </div>
      ) : null}

      <div className="app-body">
        {tweaks.showWorkspace ? (
          <Workspace
            projects={projects}
            projectId={projectId}
            onProjectChange={(id) => void handleProjectChange(id)}
            loading={loading}
            sessions={sessions}
            activeSession={activeSession}
            onOpenSession={(record) => void handleOpenSession(record)}
            onDeleteSession={(record) => void handleDeleteSession(record)}
            onRefreshSessions={() => projectId && void refreshSessions(projectId)}
            datasets={datasets}
            datasetFile={datasetFile}
            datasetName={datasetName}
            datasetTableName={datasetTableName}
            fileInputKey={fileInputKey}
            uploadingDataset={uploadingDataset}
            onFile={setDatasetFile}
            onDatasetName={setDatasetName}
            onDatasetTableName={setDatasetTableName}
            onUploadDataset={() => void handleUploadDataset()}
            projectName={projectName}
            projectDescription={projectDescription}
            onProjectName={setProjectName}
            onProjectDescription={setProjectDescription}
            creatingProject={creatingProject}
            onCreateProject={() => void handleCreateProject()}
          />
        ) : null}

        <div className="main">
          {activeSession ? (
            <>
              <RunHeader
                project={selectedProject}
                session={activeSession}
                datasetSummary={datasetSummary}
                stats={stats}
                streaming={streaming}
                tweaks={tweaks}
                playing={playing}
                onTogglePlay={() => setPlaying((p) => !p)}
                onView={(view) => setTweak("view", view)}
              />

              <div className="canvas-row">
                <div className="canvas">
                  {tweaks.view === "graph" ? (
                    <SwimlaneGraph
                      session={activeSession}
                      density={tweaks.density}
                      selectedIndex={selectedIndex}
                      hoverIndex={hoverIndex}
                      onSelect={setSelectedIndex}
                      onHover={setHoverIndex}
                    />
                  ) : (
                    <LinearTimeline
                      session={activeSession}
                      density={tweaks.density}
                      selectedIndex={selectedIndex}
                      hoverIndex={hoverIndex}
                      onSelect={setSelectedIndex}
                      onHover={setHoverIndex}
                    />
                  )}
                </div>

                {selectedStep ? (
                  <DetailDrawer
                    session={activeSession}
                    step={selectedStep}
                    theme="dark"
                    onClose={() => setSelectedIndex(null)}
                  />
                ) : null}
              </div>

              {activeSession.status === "waiting_for_input" && questions.length ? (
                <QuestionPanel
                  questions={questions}
                  answers={answers}
                  onAnswer={setAnswers}
                  onSubmit={() => void handleSubmitAnswers()}
                  loading={loading}
                />
              ) : null}

              {activeSession.status !== "waiting_for_input" &&
              activeSession.status !== "running" ? (
                <FollowUpBar
                  value={followUp}
                  onChange={setFollowUp}
                  onSend={() => void handleFollowUp()}
                  loading={loading}
                />
              ) : null}
            </>
          ) : (
            <EmptyCanvas
              project={selectedProject}
              datasets={datasets}
              health={health}
              goal={goal}
              onGoal={setGoal}
              onStart={() => void handleStart()}
              loading={loading}
              disabledLaunch={disabledLaunch}
            />
          )}

          {tweaks.showStream ? <LiveStream session={activeSession} /> : null}
        </div>
      </div>

      {tweaksOpen ? (
        <TweaksPanel
          tweaks={tweaks}
          onChange={setTweak}
          onClose={() => setTweaksOpen(false)}
        />
      ) : null}
      <div className="tweaks-toggle">
        <Btn
          variant="secondary"
          size="sm"
          icon={Settings2}
          onClick={() => setTweaksOpen((open) => !open)}
        >
          tweaks
        </Btn>
      </div>
    </div>
  );
}

function AppHeader({ project, session, streaming, loading, onRefresh, onTweaks, notebookHref }) {
  return (
    <div className="app-header">
      <div className="app-header__logo">
        <svg width="22" height="22" viewBox="0 0 56 56" fill="none">
          <path d="M14 12 L4 12 L4 44 L14 44" stroke="currentColor" strokeWidth="2.5" fill="none" />
          <path d="M42 12 L52 12 L52 44 L42 44" stroke="currentColor" strokeWidth="2.5" fill="none" />
          <rect x="14" y="22" width="12" height="12" fill="#6366F1" />
          <rect x="30" y="22" width="12" height="12" fill="#06D7E8" />
        </svg>
        <span className="app-header__brand">
          aiml<span className="app-header__brand-sub">/autopilot</span>
        </span>
      </div>

      <div className="app-header__crumb">
        <span>{project?.name ?? "no project"}</span>
        <ChevronRight size={12} strokeWidth={1.5} style={{ color: "var(--fg-4)" }} />
        <span>sessions</span>
        <ChevronRight size={12} strokeWidth={1.5} style={{ color: "var(--fg-4)" }} />
        <span className="app-header__crumb-current">{session?.session_id ?? "—"}</span>
      </div>

      <div className="app-header__spacer" />

      {session ? (
        <Pill
          tone={
            session.status === "running"
              ? "running"
              : session.status === "complete"
                ? "success"
                : session.status === "waiting_for_input"
                  ? "warn"
                  : session.status === "error"
                    ? "error"
                    : "neutral"
          }
          dot={session.status === "running" ? "running" : undefined}
          pulse={session.status === "running"}
        >
          {streaming ? "live · " : ""}
          {statusText(session.status)}
        </Pill>
      ) : (
        <Pill tone="neutral" dot="neutral">
          idle
        </Pill>
      )}

      <div className="app-header__divider" />

      <Btn variant="ghost" size="sm" icon={Search} kbd="⌘K">
        find
      </Btn>
      {notebookHref ? (
        <a
          className="btn btn--ghost btn--sm"
          href={notebookHref}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Download size={12} strokeWidth={1.75} />
          notebook.ipynb
        </a>
      ) : null}
      <Btn variant="ghost" size="sm" icon={RefreshCw} onClick={onRefresh} disabled={loading}>
        refresh
      </Btn>
      <IconBtn icon={Settings2} label="Tweaks" onClick={onTweaks} />
      <div className="app-header__avatar">YH</div>
    </div>
  );
}

function Workspace({
  projects,
  projectId,
  onProjectChange,
  loading,
  sessions,
  activeSession,
  onOpenSession,
  onDeleteSession,
  onRefreshSessions,
  datasets,
  datasetFile,
  datasetName,
  datasetTableName,
  fileInputKey,
  uploadingDataset,
  onFile,
  onDatasetName,
  onDatasetTableName,
  onUploadDataset,
  projectName,
  projectDescription,
  onProjectName,
  onProjectDescription,
  creatingProject,
  onCreateProject
}) {
  return (
    <aside className="workspace">
      <div className="workspace__section">
        <span className="eyebrow">Project</span>
        <select
          className="field"
          value={projectId}
          onChange={(event) => onProjectChange(event.target.value)}
          disabled={loading || projects.length === 0}
        >
          {projects.length === 0 ? <option>No projects found</option> : null}
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
      </div>

      <div className="workspace__section">
        <span className="eyebrow">New project</span>
        <input
          className="field"
          value={projectName}
          onChange={(event) => onProjectName(event.target.value)}
          placeholder="Project name"
          disabled={loading}
        />
        <textarea
          className="field"
          value={projectDescription}
          onChange={(event) => onProjectDescription(event.target.value)}
          placeholder="Description"
          disabled={loading}
        />
        <Btn
          variant="secondary"
          size="sm"
          icon={creatingProject ? Loader2 : FolderPlus}
          onClick={onCreateProject}
          disabled={loading || !projectName.trim()}
        >
          create
        </Btn>
      </div>

      <div className="workspace__section">
        <div className="workspace__title">
          <span className="eyebrow">Dataset upload</span>
          <span className="eyebrow" style={{ color: "var(--fg-3)" }}>
            {datasets.length}
          </span>
        </div>
        <label
          className="field"
          style={{
            cursor: !projectId ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            gap: 8,
            color: "var(--fg-2)"
          }}
        >
          <Upload size={14} strokeWidth={1.75} />
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {datasetFile?.name ?? "Choose file"}
          </span>
          <input
            key={fileInputKey}
            type="file"
            accept=".csv,.xlsx,.xls,.json,.db,.sqlite,.sqlite3"
            disabled={!projectId}
            onChange={(event) => onFile(event.target.files?.[0] ?? null)}
            style={{ display: "none" }}
          />
        </label>
        <input
          className="field"
          value={datasetName}
          onChange={(event) => onDatasetName(event.target.value)}
          placeholder="Dataset name"
          disabled={!projectId}
        />
        <input
          className="field"
          value={datasetTableName}
          onChange={(event) => onDatasetTableName(event.target.value)}
          placeholder="SQLite table"
          disabled={!projectId}
        />
        <Btn
          variant="secondary"
          size="sm"
          icon={uploadingDataset ? Loader2 : Upload}
          onClick={onUploadDataset}
          disabled={!projectId || !datasetFile || uploadingDataset}
        >
          upload
        </Btn>
      </div>

      <div className="workspace__section">
        <div className="workspace__title">
          <span className="eyebrow">Sessions</span>
          <IconBtn icon={RefreshCw} label="Refresh sessions" size={24} onClick={onRefreshSessions} />
        </div>
        <div className="session-list">
          {sessions.map((record) => (
            <div
              key={record.session_id}
              className={`session-row${activeSession?.session_id === record.session_id ? " is-selected" : ""}`}
            >
              <button className="session-row__open" onClick={() => onOpenSession(record)}>
                <span className={`status-dot status-dot--${record.status}`} />
                <span className="session-row__info">
                  <strong>{record.title || record.session_id}</strong>
                  <small>
                    {record.step_count} steps · {formatDate(record.updated_at)}
                  </small>
                </span>
              </button>
              <IconBtn
                icon={Trash2}
                label="Delete session"
                danger
                size={26}
                onClick={() => onDeleteSession(record)}
                disabled={record.status === "running"}
              />
            </div>
          ))}
          {sessions.length === 0 ? <p className="empty-note">No saved sessions</p> : null}
        </div>
      </div>
    </aside>
  );
}

function RunHeader({ project, session, datasetSummary, stats, streaming, tweaks, playing, onTogglePlay, onView }) {
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

function ViewToggle({ value, onChange }) {
  const options = [
    { id: "graph", icon: Workflow, label: "Graph", kbd: "G" },
    { id: "timeline", icon: ListOrdered, label: "Timeline", kbd: "T" }
  ];
  return (
    <div className="view-toggle">
      <div
        className="view-toggle__thumb"
        style={{ left: value === "graph" ? 3 : "calc(50% + 1px)" }}
      />
      {options.map((opt) => {
        const Icon = opt.icon;
        const active = value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            className={`view-toggle__btn${active ? " is-active" : ""}`}
            onClick={() => onChange(opt.id)}
            title={`Switch to ${opt.label} view`}
          >
            <Icon size={13} strokeWidth={1.75} />
            {opt.label}
            <span className="view-toggle__btn-kbd">{opt.kbd}</span>
          </button>
        );
      })}
    </div>
  );
}

function EmptyCanvas({ project, datasets, health, goal, onGoal, onStart, loading, disabledLaunch }) {
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

function QuestionPanel({ questions, answers, onAnswer, onSubmit, loading }) {
  return (
    <section className="questions-panel">
      <div className="section__title">
        <span className="section__title-text">input needed</span>
      </div>
      {questions.map((question, index) => (
        <div key={`${question.question}-${index}`} className="question-item">
          <p>{question.question}</p>
          {question.explanation ? (
            <p className="question-item__explain">{question.explanation}</p>
          ) : null}
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
            className="field"
            value={answers[index] ?? ""}
            onChange={(event) => {
              const next = [...answers];
              next[index] = event.target.value;
              onAnswer(next);
            }}
          />
        </div>
      ))}
      <div style={{ marginTop: 10 }}>
        <Btn variant="primary" size="md" icon={loading ? Loader2 : Send} onClick={onSubmit} disabled={loading}>
          submit answers
        </Btn>
      </div>
    </section>
  );
}

function FollowUpBar({ value, onChange, onSend, loading }) {
  return (
    <div style={{ borderTop: "1px solid var(--ink-700)", padding: "10px 16px", display: "flex", gap: 8, alignItems: "center", background: "var(--ink-850)" }}>
      <textarea
        className="field"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Try another tuning round focused on recall..."
        style={{ minHeight: 38, flex: 1 }}
      />
      <Btn variant="primary" size="md" icon={loading ? Loader2 : Send} onClick={onSend} disabled={loading || !value.trim()}>
        send
      </Btn>
    </div>
  );
}

function LiveStream({ session }) {
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

function TweaksPanel({ tweaks, onChange, onClose }) {
  return (
    <div className="tweaks">
      <div className="tweaks__head">
        <b>Tweaks</b>
        <IconBtn icon={X} label="Close" size={22} onClick={onClose} />
      </div>
      <div className="tweaks__body">
        <div className="tweaks__section">View</div>
        <SegRow
          label="Layout"
          value={tweaks.view}
          options={[
            { value: "graph", label: "Graph" },
            { value: "timeline", label: "Timeline" }
          ]}
          onChange={(v) => onChange("view", v)}
        />
        <SegRow
          label="Density"
          value={tweaks.density}
          options={[
            { value: "compact", label: "Compact" },
            { value: "comfortable", label: "Comfortable" }
          ]}
          onChange={(v) => onChange("density", v)}
        />
        <div className="tweaks__section">Run</div>
        <div className="tweaks__row">
          <span className="tweaks__label">Replay speed</span>
          <select
            className="tweaks__select"
            value={tweaks.replaySpeed}
            onChange={(event) => onChange("replaySpeed", event.target.value)}
          >
            {["1×", "2×", "10×", "live"].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>
        <Toggle label="Show workspace" value={tweaks.showWorkspace} onChange={(v) => onChange("showWorkspace", v)} />
        <Toggle label="Show agent legend" value={tweaks.showLegend} onChange={(v) => onChange("showLegend", v)} />
        <Toggle label="Show live stream" value={tweaks.showStream} onChange={(v) => onChange("showStream", v)} />
      </div>
    </div>
  );
}

function SegRow({ label, value, options, onChange }) {
  return (
    <div className="tweaks__row">
      <span className="tweaks__label">{label}</span>
      <div className="tweaks__seg">
        {options.map((opt) => (
          <button
            key={opt.value}
            type="button"
            className={value === opt.value ? "is-on" : ""}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <div className="tweaks__row">
      <span className="tweaks__label">{label}</span>
      <button
        type="button"
        className={`tweaks__toggle${value ? " is-on" : ""}`}
        aria-pressed={value}
        onClick={() => onChange(!value)}
      >
        <i />
      </button>
    </div>
  );
}
