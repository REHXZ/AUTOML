import { ListOrdered, Workflow } from "lucide-react";

const VIEW_OPTIONS = [
  { id: "graph", icon: Workflow, label: "Graph", kbd: "G" },
  { id: "timeline", icon: ListOrdered, label: "Timeline", kbd: "T" }
];

export default function ViewToggle({ value, onChange }) {
  return (
    <div className="view-toggle">
      <div
        className="view-toggle__thumb"
        style={{ left: value === "graph" ? 3 : "calc(50% + 1px)" }}
      />
      {VIEW_OPTIONS.map((opt) => {
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
