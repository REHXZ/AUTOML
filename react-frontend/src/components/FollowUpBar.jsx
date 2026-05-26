import { Loader2, Play, Plus, Send } from "lucide-react";

import { Btn } from "../ui";

export default function FollowUpBar({ value, onChange, onSend, onResume, onNewRun, isPaused, loading }) {
  const handleResume = () => {
    onResume(value.trim() || undefined);
  };

  return (
    <div style={{ borderTop: "1px solid var(--ink-700)", padding: "10px 16px", display: "flex", gap: 8, alignItems: "center", background: "var(--ink-850)" }}>
      {isPaused ? (
        <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--warning, #f59e0b)", fontSize: 12, fontWeight: 600, whiteSpace: "nowrap" }}>
          <Play size={11} />
          paused
        </div>
      ) : null}
      <textarea
        className="field"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={
          isPaused
            ? "Steer the agent, or leave blank to resume where it left off…"
            : "Try another tuning round focused on recall…"
        }
        style={{ minHeight: 38, flex: 1 }}
      />
      {isPaused ? (
        <>
          <Btn
            variant="secondary"
            size="md"
            icon={loading ? Loader2 : Play}
            onClick={handleResume}
            disabled={loading}
          >
            resume
          </Btn>
          {value.trim() ? (
            <Btn
              variant="primary"
              size="md"
              icon={loading ? Loader2 : Send}
              onClick={() => onResume(value.trim())}
              disabled={loading}
            >
              steer
            </Btn>
          ) : null}
          <Btn
            variant="secondary"
            size="md"
            icon={Plus}
            onClick={onNewRun}
            disabled={loading}
          >
            new run
          </Btn>
        </>
      ) : (
        <>
          <Btn variant="primary" size="md" icon={loading ? Loader2 : Send} onClick={onSend} disabled={loading || !value.trim()}>
            send
          </Btn>
          <Btn variant="secondary" size="md" icon={Plus} onClick={onNewRun} disabled={loading}>
            new run
          </Btn>
        </>
      )}
    </div>
  );
}
