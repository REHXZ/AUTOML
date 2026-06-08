import { Eye, EyeOff, X } from "lucide-react";
import { useState } from "react";

import { IconBtn } from "../ui";

export default function TweaksPanel({ tweaks, onChange, apiKey, onApiKey, onClose }) {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="tweaks">
      <div className="tweaks__head">
        <b>Tweaks</b>
        <IconBtn icon={X} label="Close" size={22} onClick={onClose} />
      </div>
      <div className="tweaks__body">
        <div className="tweaks__section">API</div>
        <div className="tweaks__row" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
          <span className="tweaks__label">OpenAI API Key</span>
          <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
            <input
              type={showKey ? "text" : "password"}
              value={apiKey ?? ""}
              onChange={(e) => onApiKey(e.target.value)}
              placeholder="sk-…"
              style={{
                flex: 1,
                background: "var(--ink-800)",
                border: "1px solid var(--ink-600)",
                borderRadius: 4,
                padding: "4px 30px 4px 8px",
                color: "var(--fg-1)",
                fontSize: 12,
                fontFamily: "monospace",
                outline: "none",
              }}
              autoComplete="off"
              spellCheck={false}
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              style={{ position: "absolute", right: 6, background: "none", border: "none", cursor: "pointer", color: "var(--fg-3)", display: "flex", alignItems: "center" }}
              tabIndex={-1}
              aria-label={showKey ? "Hide" : "Show"}
            >
              {showKey ? <EyeOff size={12} /> : <Eye size={12} />}
            </button>
          </div>
          {apiKey?.trim() ? (
            <span style={{ fontSize: 10, color: "var(--fg-4)" }}>Saved in browser only · never sent to any server except OpenAI</span>
          ) : (
            <span style={{ fontSize: 10, color: "var(--warning, #f59e0b)" }}>Required to launch runs</span>
          )}
        </div>

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
