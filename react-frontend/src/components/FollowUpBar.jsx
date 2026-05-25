import { Loader2, Send } from "lucide-react";

import { Btn } from "../ui";

export default function FollowUpBar({ value, onChange, onSend, loading }) {
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
