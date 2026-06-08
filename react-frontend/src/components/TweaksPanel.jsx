import { Eye, EyeOff, X } from "lucide-react";
import { useState } from "react";

import { IconBtn } from "../ui";

const PROVIDERS = [
  { value: "auto",      label: "Auto-detect" },
  { value: "openai",    label: "OpenAI" },
  { value: "azure",     label: "Azure OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "ollama",    label: "Ollama (local)" },
  { value: "custom",    label: "Custom endpoint" },
];

const PROVIDER_DEFAULTS = {
  openai:    { placeholder: "sk-…", modelPlaceholder: "gpt-4o", needsKey: true, needsBase: false },
  azure:     { placeholder: "Azure API key", modelPlaceholder: "gpt-4o (deployment name)", needsKey: true, needsBase: true, baseLabel: "Azure endpoint" },
  anthropic: { placeholder: "sk-ant-…", modelPlaceholder: "claude-opus-4-8", needsKey: true, needsBase: false },
  ollama:    { placeholder: "(none required)", modelPlaceholder: "llama3", needsKey: false, needsBase: true, baseLabel: "Ollama URL" },
  custom:    { placeholder: "API key", modelPlaceholder: "model name", needsKey: true, needsBase: true, baseLabel: "Base URL" },
  auto:      { placeholder: "API key (if not in env)", modelPlaceholder: "override model", needsKey: true, needsBase: false },
};

export default function TweaksPanel({ tweaks, onChange, providerConfig, onProviderConfig, serverProviders, onClose }) {
  const [showKey, setShowKey] = useState(false);

  const cfg = providerConfig ?? { provider: "auto", api_key: "", model: "", base_url: "", api_version: "2024-12-01-preview" };
  const provider = cfg.provider || "auto";
  const defs = PROVIDER_DEFAULTS[provider] ?? PROVIDER_DEFAULTS.auto;
  const autoDetected = serverProviders?.auto_detected;
  const configured = serverProviders?.configured ?? [];

  const update = (field, value) => onProviderConfig?.({ ...cfg, [field]: value });

  return (
    <div className="tweaks">
      <div className="tweaks__head">
        <b>Tweaks</b>
        <IconBtn icon={X} label="Close" size={22} onClick={onClose} />
      </div>
      <div className="tweaks__body">

        {/* ── Provider ── */}
        <div className="tweaks__section">LLM Provider</div>

        {autoDetected && (
          <div style={{ fontSize: 10, color: "var(--fg-3)", padding: "0 0 6px 0" }}>
            Auto-detected from env: <b style={{ color: "var(--fg-2)" }}>{autoDetected}</b>
            {configured.length > 1 && ` (also: ${configured.filter(p => p !== autoDetected).join(", ")})`}
          </div>
        )}

        {/* Provider selector */}
        <div className="tweaks__row" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
          <span className="tweaks__label">Provider</span>
          <select
            className="tweaks__select"
            style={{ width: "100%", background: "var(--ink-800)", color: "var(--fg-1)", border: "1px solid var(--ink-600)", borderRadius: 4, padding: "4px 8px", fontSize: 12 }}
            value={provider}
            onChange={(e) => update("provider", e.target.value)}
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
                {configured.includes(p.value) ? " ✓" : ""}
                {p.value === autoDetected ? " (auto)" : ""}
              </option>
            ))}
          </select>
        </div>

        {/* API Key */}
        {defs.needsKey && (
          <div className="tweaks__row" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
            <span className="tweaks__label">API Key</span>
            <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
              <input
                type={showKey ? "text" : "password"}
                value={cfg.api_key ?? ""}
                onChange={(e) => update("api_key", e.target.value)}
                placeholder={defs.placeholder}
                style={inputStyle}
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                onClick={() => setShowKey((v) => !v)}
                style={eyeStyle}
                tabIndex={-1}
                aria-label={showKey ? "Hide" : "Show"}
              >
                {showKey ? <EyeOff size={12} /> : <Eye size={12} />}
              </button>
            </div>
            {cfg.api_key?.trim() ? (
              <span style={{ fontSize: 10, color: "var(--fg-4)" }}>Stored in browser only</span>
            ) : configured.includes(provider) ? (
              <span style={{ fontSize: 10, color: "var(--accent-green, #22c55e)" }}>Using env variable</span>
            ) : (
              <span style={{ fontSize: 10, color: "var(--warning, #f59e0b)" }}>Required to launch runs</span>
            )}
          </div>
        )}

        {/* Base URL (Azure endpoint / Ollama URL / custom) */}
        {defs.needsBase && (
          <div className="tweaks__row" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
            <span className="tweaks__label">{defs.baseLabel ?? "Base URL"}</span>
            <input
              type="text"
              value={cfg.base_url ?? ""}
              onChange={(e) => update("base_url", e.target.value)}
              placeholder={provider === "azure" ? "https://<resource>.openai.azure.com/" : provider === "ollama" ? "http://localhost:11434/v1" : "https://…"}
              style={{ ...inputStyle, fontFamily: "monospace", fontSize: 11 }}
              autoComplete="off"
              spellCheck={false}
            />
          </div>
        )}

        {/* Azure API version */}
        {provider === "azure" && (
          <div className="tweaks__row" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
            <span className="tweaks__label">API Version</span>
            <input
              type="text"
              value={cfg.api_version ?? ""}
              onChange={(e) => update("api_version", e.target.value)}
              placeholder="2024-12-01-preview"
              style={{ ...inputStyle, fontFamily: "monospace", fontSize: 11 }}
            />
          </div>
        )}

        {/* Model override */}
        <div className="tweaks__row" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
          <span className="tweaks__label">Model <span style={{ fontWeight: 400, color: "var(--fg-4)" }}>(optional override)</span></span>
          <input
            type="text"
            value={cfg.model ?? ""}
            onChange={(e) => update("model", e.target.value)}
            placeholder={defs.modelPlaceholder}
            style={{ ...inputStyle, fontFamily: "monospace", fontSize: 11 }}
          />
        </div>

        {/* ── View ── */}
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

        {/* ── Run ── */}
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

const inputStyle = {
  flex: 1,
  width: "100%",
  background: "var(--ink-800)",
  border: "1px solid var(--ink-600)",
  borderRadius: 4,
  padding: "4px 30px 4px 8px",
  color: "var(--fg-1)",
  fontSize: 12,
  fontFamily: "inherit",
  outline: "none",
};

const eyeStyle = {
  position: "absolute",
  right: 6,
  background: "none",
  border: "none",
  cursor: "pointer",
  color: "var(--fg-3)",
  display: "flex",
  alignItems: "center",
};

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
