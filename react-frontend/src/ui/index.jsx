export function AgentDot({ color, size = 10, glow = false, pulse = false, style }) {
  return (
    <span
      className={`agent-dot${pulse ? " agent-dot--pulse" : ""}`}
      style={{
        width: size,
        height: size,
        background: color,
        boxShadow: glow ? `0 0 12px ${color}, 0 0 0 1px ${color}40` : "none",
        ...style
      }}
    />
  );
}

export function Kbd({ children }) {
  return <span className="kbd">{children}</span>;
}

export function Pill({ tone = "neutral", dot, pulse, icon: IconCmp, children }) {
  return (
    <span className={`pill pill--${tone}`}>
      {dot ? (
        <span
          className={`pill__dot pill__dot--${dot}${pulse ? " pill__dot--pulse" : ""}`}
        />
      ) : null}
      {IconCmp ? <IconCmp size={11} strokeWidth={1.75} /> : null}
      {children}
    </span>
  );
}

export function Btn({
  variant = "secondary",
  size = "md",
  icon: IconCmp,
  kbd,
  disabled = false,
  onClick,
  type = "button",
  children,
  style,
  ...rest
}) {
  return (
    <button
      type={type}
      className={`btn btn--${variant} btn--${size}`}
      onClick={onClick}
      disabled={disabled}
      style={style}
      {...rest}
    >
      {IconCmp ? <IconCmp size={size === "sm" ? 12 : 13} strokeWidth={1.75} /> : null}
      {children}
      {kbd ? <Kbd>{kbd}</Kbd> : null}
    </button>
  );
}

export function IconBtn({
  icon: IconCmp,
  label,
  onClick,
  active = false,
  size = 28,
  kbd,
  danger = false,
  disabled = false,
  style
}) {
  const cls = ["icon-btn", active ? "is-active" : "", danger ? "is-danger" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <button
      type="button"
      className={cls}
      onClick={onClick}
      disabled={disabled}
      title={label + (kbd ? ` (${kbd})` : "")}
      style={{ width: size, height: size, ...style }}
    >
      <IconCmp size={Math.round(size * 0.55)} strokeWidth={1.75} />
    </button>
  );
}
