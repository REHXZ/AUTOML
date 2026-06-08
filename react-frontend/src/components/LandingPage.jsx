import { useEffect, useRef } from "react";
import "../styles/landing.css";
import AutopilotShowcase from "./AutopilotShowcase";

const AGENTS = [
  { id: "01", icon: "psychology",    name: "Scientist",           desc: "Process Orchestration" },
  { id: "02", icon: "bar_chart",     name: "EDA",                 desc: "Visual Insights & Data Profiling" },
  { id: "03", icon: "transform",     name: "Feature Engineering", desc: "Data Transformation & Selection" },
  { id: "04", icon: "memory",        name: "Modeling",            desc: "AutoML Engine (25+ Models)" },
  { id: "05", icon: "fact_check",    name: "Review",              desc: "Quality Control & Validation" },
  { id: "06", icon: "tune",          name: "Fine Tuning",         desc: "Optuna Hyperparameter Opt" },
  { id: "07", icon: "monitor_heart", name: "Drift",               desc: "Data Distribution Observability & Alerting Mechanisms", wide: true },
];

const FEATURES = [
  { icon: "lock",   text: "Local-First Architecture" },
  { icon: "loop",   text: "Extensible Hook System" },
  { icon: "book_4", text: "CRISP-DM Notebook Export" },
  { icon: "bolt",   text: "25+ Optimized Models" },
];

export default function LandingPage({ onEnterApp }) {
  const rootRef = useRef(null);

  useEffect(() => {
    // Inject Material Symbols font if not already present
    if (!document.getElementById("lp-material-symbols")) {
      const link = document.createElement("link");
      link.id = "lp-material-symbols";
      link.rel = "stylesheet";
      link.href =
        "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200";
      document.head.appendChild(link);
    }

    // Allow scrolling while landing page is shown (app sets overflow:hidden)
    const prevBodyOverflow = document.body.style.overflow;
    const prevHtmlOverflow = document.documentElement.style.overflow;
    document.body.style.overflow = "auto";
    document.documentElement.style.overflow = "auto";

    // Fade-in on scroll
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { root: null, rootMargin: "0px", threshold: 0.1 }
    );

    rootRef.current?.querySelectorAll(".lp-fade-in").forEach((el) => observer.observe(el));

    return () => {
      observer.disconnect();
      document.body.style.overflow = prevBodyOverflow;
      document.documentElement.style.overflow = prevHtmlOverflow;
    };
  }, []);

  return (
    <div className="lp-root" ref={rootRef}>
      <div className="lp-scanline" aria-hidden="true" />
      <div className="lp-dotgrid" aria-hidden="true" />

      {/* ---- Header ---- */}
      <header className="lp-header">
        <div className="lp-header-inner">
          <div className="lp-logo">AUTOML</div>

          <nav className="lp-nav" aria-label="Main navigation">
            <a className="lp-nav-link lp-nav-link--active" href="#">Platform</a>
            <a className="lp-nav-link" href="#">Core Engine</a>
            <a className="lp-nav-link" href="#">Agent Showcase</a>
            <a className="lp-nav-link" href="#">Documentation</a>
          </nav>

          <div className="lp-header-actions">
            <a className="lp-btn-ghost" href="https://github.com/REHXZ/AUTOML" target="_blank" rel="noreferrer">GitHub</a>
            <button className="lp-btn-primary" onClick={onEnterApp}>
              Initialize Terminal
            </button>
          </div>

          <button className="lp-hamburger" aria-label="Open menu">
            <span className="material-symbols-outlined">menu</span>
          </button>
        </div>
      </header>

      <main className="lp-main">
        {/* ---- Hero ---- */}
        <section className="lp-section lp-hero-section lp-fade-in">
          <h1 className="lp-hero-title">Autonomous ML Discovery. Local-First.</h1>
          <p className="lp-hero-desc">
            Orchestrate a swarm of AI agents to handle the full ML lifecycle. From CSV to
            CRISP-DM structured Jupyter notebooks, without your data ever leaving your machine.
          </p>
          <div className="lp-hero-actions">
            <a
              className="lp-btn-cta-primary"
              href="https://github.com/REHXZ/AUTOML"
              target="_blank"
              rel="noreferrer"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>terminal</span>
              View on GitHub
            </a>
            <a
              className="lp-btn-cta-secondary"
              href="https://github.com/REHXZ/AUTOML/blob/main/README.md"
              target="_blank"
              rel="noreferrer"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>book</span>
              Read the Docs
            </a>
          </div>
        </section>

        {/* ---- Execution Flow ---- */}
        <section className="lp-section lp-fade-in">
          <div className="lp-section-header">
            <span className="lp-badge">SYS.PROCEDURE</span>
            <h2 className="lp-section-title">Execution Flow</h2>
          </div>
          <div className="lp-how-grid">
            {[
              { n: "01", icon: "upload_file", title: "Connect Data",     desc: "Initialize the environment by connecting your local CSV datasets securely." },
              { n: "02", icon: "target",      title: "Define Objective", desc: "Provide a plain English goal. The Scientist agent interprets and plans the task." },
              { n: "03", icon: "play_arrow",  title: "Execute & Export", desc: "The swarm executes the pipeline, outputting trained models and a structured Jupyter Notebook." },
            ].map((step) => (
              <div key={step.n} className="lp-how-card">
                <div className="lp-card-accent" aria-hidden="true" />
                <div className="lp-how-number">{step.n}</div>
                <div className="lp-how-icon-row">
                  <span className="material-symbols-outlined lp-how-card-icon">{step.icon}</span>
                  <h3 className="lp-how-card-title">{step.title}</h3>
                </div>
                <p className="lp-how-card-desc">{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ---- Agent Swarm ---- */}
        <section className="lp-section lp-fade-in">
          <div className="lp-section-header">
            <span className="lp-badge">SYS.AGENTS</span>
            <h2 className="lp-section-title">Active Swarm Topology</h2>
          </div>
          <div className="lp-agent-grid">
            {AGENTS.map((agent) => (
              <div
                key={agent.id}
                className={`lp-agent-card${agent.wide ? " lp-agent-card--wide" : ""}`}
              >
                <div className="lp-agent-card-header">
                  <span className="material-symbols-outlined lp-agent-icon">{agent.icon}</span>
                  <span className="lp-agent-id">ID:{agent.id}</span>
                </div>
                <h4 className="lp-agent-name">{agent.name}</h4>
                <p className="lp-agent-desc">{agent.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ---- Autopilot in Action ---- */}
        <section className="lp-section lp-fade-in">
          <div className="lp-section-header">
            <span className="lp-badge">SYS.AUTOPILOT</span>
            <h2 className="lp-section-title">Autopilot in Action</h2>
          </div>
          <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#849495", marginBottom: 28, marginTop: -24, letterSpacing: "0.04em" }}>
            The Scientist orchestrates up to 60 iterations, delegating to specialist agents in real time.
            Cyan pulses show delegation; gray pulses show results flowing back.
          </p>
          <AutopilotShowcase />
        </section>

        {/* ---- Core Capabilities + Quick Start ---- */}
        <section className="lp-section lp-section--no-border lp-fade-in">
          <div className="lp-features-grid">
            {/* Left: capabilities */}
            <div className="lp-features-col">
              <div className="lp-section-header">
                <span className="lp-badge">SYS.SPECS</span>
                <h2 className="lp-section-title">Core Capabilities</h2>
              </div>
              <ul className="lp-features-list">
                {FEATURES.map((f) => (
                  <li key={f.text} className="lp-feature-item">
                    <span className="material-symbols-outlined lp-feature-icon">{f.icon}</span>
                    <span className="lp-feature-text">{f.text}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Right: terminal */}
            <div className="lp-terminal-col">
              <div className="lp-section-header">
                <span className="lp-badge">SYS.INIT</span>
                <h2 className="lp-section-title">Quick Start</h2>
              </div>
              <div className="lp-terminal">
                <div className="lp-terminal-bar">
                  <div className="lp-terminal-dot" />
                  <div className="lp-terminal-dot" />
                  <div className="lp-terminal-dot" />
                  <span className="lp-terminal-label">bash — aiml-discovery</span>
                </div>
                <div className="lp-terminal-body">
                  <div className="lp-terminal-line">
                    <span className="lp-terminal-prompt">$</span>
                    <span className="lp-terminal-cmd">pip install aiml-discovery</span>
                  </div>
                  <div className="lp-terminal-line">
                    <span className="lp-terminal-prompt">$</span>
                    <span className="lp-terminal-cmd">aiml-discovery init</span>
                  </div>
                  <div className="lp-terminal-line">
                    <span className="lp-terminal-prompt">$</span>
                    <span className="lp-terminal-cmd lp-terminal-cmd--accent">
                      aiml-discovery run --data ./data.csv --goal &quot;Predict churn&quot;
                    </span>
                  </div>
                  <div className="lp-terminal-line lp-terminal-output">
                    <span>&gt; Swarm Initialized. Orchestrating pipeline...</span>
                  </div>
                  <div className="lp-terminal-line lp-terminal-output">
                    <span>&gt; Tasks complete. Models saved to ./output</span>
                  </div>
                  <div className="lp-terminal-line">
                    <span className="lp-terminal-prompt">$</span>
                    <span className="lp-terminal-cmd">
                      jupyter notebook ./output/pipeline_report.ipynb
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

      </main>

      {/* ---- Footer ---- */}
      <footer className="lp-footer">
        <div className="lp-footer-inner">
          <div className="lp-footer-brand">AUTOML</div>
          <div className="lp-footer-copy">
            © 2026 AUTOML — LICENSED UNDER MIT OPEN SOURCE
          </div>
          <nav className="lp-footer-nav" aria-label="Footer links">
            <a className="lp-footer-link" href="https://github.com/REHXZ/AUTOML" target="_blank" rel="noreferrer">GitHub Repository</a>
            <a className="lp-footer-link" href="#">Security Protocol</a>
            <a className="lp-footer-link" href="#">Technical Specs</a>
            <a className="lp-footer-link" href="#">Privacy Log</a>
          </nav>
        </div>
      </footer>
    </div>
  );
}
