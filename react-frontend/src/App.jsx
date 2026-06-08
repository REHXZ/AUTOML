import { AlertTriangle, Settings2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AgentOutputPanel from "./components/AgentOutputPanel";
import AppHeader from "./components/AppHeader";
import DetailDrawer from "./components/DetailDrawer";
import EmptyCanvas from "./components/EmptyCanvas";
import FollowUpBar from "./components/FollowUpBar";
import LinearTimeline from "./components/LinearTimeline";
import LiveStream from "./components/LiveStream";
import ModelReviewPage from "./components/ModelReviewPage";
import QuestionPanel from "./components/QuestionPanel";
import RunHeader from "./components/RunHeader";
import SwimlaneGraph from "./components/SwimlaneGraph";
import TweaksPanel from "./components/TweaksPanel";
import Workspace from "./components/Workspace";
import {
  connectSessionEvents,
  createProject,
  deleteSession,
  getHealth,
  getProviders,
  getSession,
  listDatasets,
  listProjects,
  listRuns,
  listSessions,
  notebookUrl,
  sendFollowUp,
  startSession,
  stopSession,
  submitAnswers,
  uploadDataset
} from "./api";
import { Btn } from "./ui";
import {
  getSessionStats,
  isTerminalStatus,
  maxStepIndex,
  mergeStep,
  normalizeQuestions,
  visibleActivitySteps
} from "./utils";

const PROVIDER_CFG_STORAGE = "aiml-provider-config";
// Legacy key — migrated to PROVIDER_CFG_STORAGE on first load
const LEGACY_API_KEY_STORAGE = "aiml-autopilot-apikey";

const DEFAULT_PROVIDER_CONFIG = {
  provider: "auto",
  api_key: "",
  model: "",
  base_url: "",
  api_version: "2024-12-01-preview",
};

function loadProviderConfig() {
  try {
    const raw = window.localStorage?.getItem(PROVIDER_CFG_STORAGE);
    if (raw) return { ...DEFAULT_PROVIDER_CONFIG, ...JSON.parse(raw) };
    // Migrate legacy bare API key
    const legacyKey = window.localStorage?.getItem(LEGACY_API_KEY_STORAGE) || "";
    if (legacyKey) return { ...DEFAULT_PROVIDER_CONFIG, api_key: legacyKey };
  } catch { /* ignore */ }
  return DEFAULT_PROVIDER_CONFIG;
}

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

export default function App({ user, onSignOut }) {
  const [providerConfig, setProviderConfigState] = useState(loadProviderConfig);
  const [serverProviders, setServerProviders] = useState(null);

  const setProviderConfig = (cfg) => {
    const next = { ...DEFAULT_PROVIDER_CONFIG, ...cfg };
    setProviderConfigState(next);
    try { window.localStorage?.setItem(PROVIDER_CFG_STORAGE, JSON.stringify(next)); } catch { /* ignore */ }
  };

  // Backward-compat: apiKey derived from providerConfig
  const apiKey = providerConfig.api_key;

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
  const [stopping, setStopping] = useState(false);
  const [tweaks, setTweaks] = useState(loadTweaks);
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [hoverIndex, setHoverIndex] = useState(null);
  const [currentPage, setCurrentPage] = useState("autopilot");
  const [runs, setRuns] = useState([]);
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
          setStopping(false);
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
          setStopping(false);
          setError("Live event stream disconnected. Refresh the session to reconnect.");
          void refreshSession(id, sessionId);
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
      const [nextHealth, nextProjects, nextProviders] = await Promise.all([
        getHealth(),
        listProjects(),
        getProviders().catch(() => null),
      ]);
      setHealth(nextHealth);
      setProjects(nextProjects);
      if (nextProviders) setServerProviders(nextProviders);
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
      const job = await startSession(projectId, goal.trim(), providerConfig);
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
      await sendFollowUp(projectId, activeSession.session_id, followUp.trim(), providerConfig);
      const loaded = await refreshSession(projectId, activeSession.session_id);
      openEventStream(projectId, activeSession.session_id, maxStepIndex(loaded.steps));
      setFollowUp("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleNewRun = () => {
    closeStream();
    setActiveSession(null);
    setFollowUp("");
    setSelectedIndex(null);
  };

  const handlePageChange = useCallback(async (page) => {
    setCurrentPage(page);
    if (page === "model-review" && projectId) {
      try {
        const projectRuns = await listRuns(projectId);
        setRuns(projectRuns);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }
  }, [projectId]);

  const handlePause = async () => {
    if (!projectId || !activeSession) return;
    setStopping(true);
    setError(null);
    try {
      await stopSession(projectId, activeSession.session_id);
      const loaded = await refreshSession(projectId, activeSession.session_id);
      // If the agent already stopped by the time the HTTP call returned, unblock immediately.
      // If still running, stopping stays true and the SSE status:idle event will clear it.
      if (loaded.status !== "running") {
        setStopping(false);
      }
    } catch (err) {
      setStopping(false);
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleResume = async (message) => {
    const text = (message || "Continue the analysis from where you left off.").trim();
    if (!projectId || !activeSession) return;
    setLoading(true);
    setError(null);
    try {
      await sendFollowUp(projectId, activeSession.session_id, text, providerConfig);
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

  const providerReady = providerConfig.provider === "ollama"
    || !!providerConfig.api_key?.trim()
    || providerConfig.provider === "auto";
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
        currentPage={currentPage}
        onPageChange={(page) => void handlePageChange(page)}
        user={user}
        onSignOut={onSignOut}
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
          {currentPage === "model-review" ? (
            <ModelReviewPage
              projectId={projectId}
              runs={runs}
              datasets={datasets}
              onBack={() => void handlePageChange("autopilot")}
            />
          ) : activeSession ? (
            <>
              <RunHeader
                project={selectedProject}
                session={activeSession}
                datasetSummary={datasetSummary}
                stats={stats}
                streaming={streaming}
                stopping={stopping}
                tweaks={tweaks}
                onPause={() => void handlePause()}
                onResume={() => void handleResume()}
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

              <AgentOutputPanel
                session={activeSession}
                onSelectStep={setSelectedIndex}
              />

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
              (activeSession.status !== "running" || stopping) ? (
                <FollowUpBar
                  value={followUp}
                  onChange={setFollowUp}
                  onSend={() => void handleFollowUp()}
                  onResume={(msg) => void handleResume(msg)}
                  onNewRun={handleNewRun}
                  isPaused={activeSession.status === "idle" || stopping}
                  loading={loading || stopping}
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
              providerConfig={providerConfig}
              onProviderConfig={setProviderConfig}
              serverProviders={serverProviders}
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
          providerConfig={providerConfig}
          onProviderConfig={setProviderConfig}
          serverProviders={serverProviders}
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
