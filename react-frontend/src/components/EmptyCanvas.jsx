import { ChevronDown, Eye, EyeOff, Loader2, Play } from "lucide-react";
import { useState } from "react";

import { Btn } from "../ui";

const PROVIDERS = [
  { value: "auto",      label: "Auto-detect" },
  { value: "openai",    label: "OpenAI" },
  { value: "azure",     label: "Azure OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "ollama",    label: "Ollama (local)" },
  { value: "custom",    label: "Custom endpoint" },
];

const PROVIDER_DEFS = {
  openai:    { needsKey: true,  needsBase: false, keyPlaceholder: "sk-…", modelPlaceholder: "gpt-4o" },
  azure:     { needsKey: true,  needsBase: true,  keyPlaceholder: "Azure API key", modelPlaceholder: "gpt-4o (deployment name)", baseLabel: "Azure endpoint", basePlaceholder: "https://<resource>.openai.azure.com/" },
  anthropic: { needsKey: true,  needsBase: false, keyPlaceholder: "sk-ant-…", modelPlaceholder: "claude-opus-4-8" },
  ollama:    { needsKey: false, needsBase: true,  baseLabel: "Ollama URL", basePlaceholder: "http://localhost:11434/v1", modelPlaceholder: "llama3" },
  custom:    { needsKey: true,  needsBase: true,  keyPlaceholder: "API key", baseLabel: "Base URL", basePlaceholder: "https://…", modelPlaceholder: "model name" },
  auto:      { needsKey: true,  needsBase: false, keyPlaceholder: "API key (if not in env)", modelPlaceholder: "override model" },
};

export default function EmptyCanvas({ project, datasets, health, goal, onGoal, providerConfig, onProviderConfig, serverProviders, onStart, loading, disabledLaunch }) {
  const [showKey, setShowKey] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const cfg = providerConfig ?? { provider: "auto", api_key: "", model: "", base_url: "", api_version: "2024-12-01-preview" };
  const provider = cfg.provider || "auto";
  const defs = PROVIDER_DEFS[provider] ?? PROVIDER_DEFS.auto;
  const autoDetected = serverProviders?.auto_detected;
  const configured = serverProviders?.configured ?? [];

  const keyMissing = defs.needsKey && !cfg.api_key?.trim() && !configured.includes(provider);
  const launchDisabled = disabledLaunch || keyMissing;

  const update = (field, value) => onProviderConfig?.({ ...cfg, [field]: value });

  const backendOk = health?.openai_configured
    || (health?.configured_providers ?? []).length > 0
    || autoDetected != null;

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

          {/* ── Provider selector ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="eyebrow" style={{ color: "var(--fg-3)" }}>LLM Provider</label>

            {autoDetected && !showAdvanced && (
              <div style={{ fontSize: 11, color: "var(--fg-3)" }}>
                Auto-detected: <b style={{ color: "var(--fg-2)" }}>{autoDetected}</b>
                {" · "}
                <button
                  type="button"
                  onClick={() => setShowAdvanced(true)}
                  style={{ background: "none", border: "none", color: "var(--accent, #6366f1)", cursor: "pointer", fontSize: 11, padding: 0 }}
                >
                  change
                </button>
              </div>
            )}

            {(!autoDetected || showAdvanced) && (
              <>
                <div style={{ position: "relative" }}>
                  <select
                    className="field"
                    style={{ width: "100%", paddingRight: 28, fontFamily: "inherit", fontSize: 13, appearance: "none" }}
                    value={provider}
                    onChange={(e) => update("provider", e.target.value)}
                  >
                    {PROVIDERS.map((p) => (
                      <option key={p.value} value={p.value}>
                        {p.label}
                        {configured.includes(p.value) ? " ✓" : ""}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={14} style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", pointerEvents: "none", color: "var(--fg-3)" }} />
                </div>

                {/* API Key */}
                {defs.needsKey && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                      <input
                        className="field"
                        type={showKey ? "text" : "password"}
                        value={cfg.api_key ?? ""}
                        onChange={(e) => update("api_key", e.target.value)}
                        placeholder={defs.keyPlaceholder}
                        style={{ flex: 1, paddingRight: 36, fontFamily: "monospace", fontSize: 13 }}
                        autoComplete="off"
                        spellCheck={false}
                      />
                      <button
                        type="button"
                        onClick={() => setShowKey((v) => !v)}
                        style={{ position: "absolute", right: 8, background: "none", border: "none", cursor: "pointer", color: "var(--fg-3)", display: "flex", alignItems: "center" }}
                        tabIndex={-1}
                      >
                        {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                      </button>
                    </div>
                    {cfg.api_key?.trim() ? (
                      <span style={{ fontSize: 11, color: "var(--fg-4)" }}>Stored in browser only</span>
                    ) : configured.includes(provider) ? (
                      <span style={{ fontSize: 11, color: "var(--accent-green, #22c55e)" }}>Using environment variable</span>
                    ) : keyMissing ? (
                      <span style={{ fontSize: 11, color: "var(--warning, #f59e0b)" }}>
                        API key required to launch a run.
                      </span>
                    ) : null}
                  </div>
                )}

                {/* Base URL */}
                {defs.needsBase && (
                  <input
                    className="field"
                    type="text"
                    value={cfg.base_url ?? ""}
                    onChange={(e) => update("base_url", e.target.value)}
                    placeholder={defs.basePlaceholder}
                    style={{ fontFamily: "monospace", fontSize: 12 }}
                    autoComplete="off"
                  />
                )}
              </>
            )}
          </div>

          {/* ── Goal ── */}
          <textarea
            className="field"
            value={goal}
            onChange={(event) => onGoal(event.target.value)}
            placeholder="Predict churn, explain revenue drivers, find the best forecasting setup..."
            disabled={launchDisabled && !disabledLaunch}
            style={{ minHeight: 96 }}
          />

          {/* ── Launch ── */}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Btn
              variant="primary"
              size="lg"
              icon={loading ? Loader2 : Play}
              onClick={onStart}
              disabled={launchDisabled}
            >
              {loading ? "starting…" : "launch run"}
            </Btn>
            <span className="eyebrow" style={{ color: "var(--fg-4)" }}>
              {datasets.length} datasets · {backendOk ? "backend ready" : "backend offline"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
