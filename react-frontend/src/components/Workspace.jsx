import {
  FolderPlus,
  Loader2,
  RefreshCw,
  Trash2,
  Upload
} from "lucide-react";

import { Btn, IconBtn } from "../ui";
import { formatDate } from "../utils";

export default function Workspace({
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
