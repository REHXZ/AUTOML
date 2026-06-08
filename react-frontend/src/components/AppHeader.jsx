import {
  ChevronRight,
  Download,
  FlaskConical,
  RefreshCw,
  Search,
  Settings2
} from "lucide-react";

import { Btn, IconBtn, Pill } from "../ui";
import { statusText } from "../utils";

export default function AppHeader({ project, session, streaming, loading, onRefresh, onTweaks, notebookHref, currentPage, onPageChange, user, onSignOut }) {
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
        <span
          style={{ cursor: currentPage === "model-review" ? "pointer" : "default", color: currentPage === "model-review" ? "var(--fg-3)" : undefined }}
          onClick={currentPage === "model-review" ? () => onPageChange("autopilot") : undefined}
        >
          {currentPage === "model-review" ? "sessions" : "sessions"}
        </span>
        {currentPage === "model-review" ? (
          <>
            <ChevronRight size={12} strokeWidth={1.5} style={{ color: "var(--fg-4)" }} />
            <span className="app-header__crumb-current">model review</span>
          </>
        ) : (
          <>
            <ChevronRight size={12} strokeWidth={1.5} style={{ color: "var(--fg-4)" }} />
            <span className="app-header__crumb-current">{session?.session_id ?? "—"}</span>
          </>
        )}
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

      <Btn
        variant={currentPage === "model-review" ? "primary" : "ghost"}
        size="sm"
        icon={FlaskConical}
        disabled={!project}
        onClick={() => onPageChange(currentPage === "model-review" ? "autopilot" : "model-review")}
      >
        model review
      </Btn>

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
      {onSignOut ? (
        <Btn variant="ghost" size="sm" onClick={onSignOut} title={user?.email}>
          sign out
        </Btn>
      ) : null}
      <div className="app-header__avatar" title={user?.email}>
        {user?.email?.[0]?.toUpperCase() ?? "U"}
      </div>
    </div>
  );
}
