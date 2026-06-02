import { X } from "lucide-react";

import { IconBtn } from "../ui";

export default function TweaksPanel({ tweaks, onChange, onClose }) {
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
