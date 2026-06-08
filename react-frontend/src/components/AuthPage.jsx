import { useState } from "react";
import { supabase } from "../lib/supabase";

export default function AuthPage({ onSuccess }) {
  const [mode, setMode] = useState("login"); // login | signup | magic
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const reset = () => { setError(null); setNotice(null); };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!supabase) return;
    reset();
    setLoading(true);
    try {
      if (mode === "login") {
        const { error: err } = await supabase.auth.signInWithPassword({ email, password });
        if (err) throw err;
        onSuccess?.();
      } else if (mode === "signup") {
        const { error: err } = await supabase.auth.signUp({ email, password });
        if (err) throw err;
        setNotice("Check your email to confirm your account, then sign in.");
        setMode("login");
      } else {
        const { error: err } = await supabase.auth.signInWithOtp({ email });
        if (err) throw err;
        setNotice("Magic link sent — check your inbox.");
      }
    } catch (err) {
      setError(err.message ?? "Authentication failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">
          <svg width="28" height="28" viewBox="0 0 56 56" fill="none">
            <path d="M14 12 L4 12 L4 44 L14 44" stroke="currentColor" strokeWidth="2.5" fill="none" />
            <path d="M42 12 L52 12 L52 44 L42 44" stroke="currentColor" strokeWidth="2.5" fill="none" />
            <rect x="14" y="22" width="12" height="12" fill="#6366F1" />
            <rect x="30" y="22" width="12" height="12" fill="#06D7E8" />
          </svg>
          <span className="auth-brand">aiml<span className="auth-brand-sub">/autopilot</span></span>
        </div>

        <div className="auth-tabs">
          {[["login", "Sign in"], ["signup", "Create account"], ["magic", "Magic link"]].map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`auth-tab${mode === key ? " auth-tab--active" : ""}`}
              onClick={() => { setMode(key); reset(); }}
            >
              {label}
            </button>
          ))}
        </div>

        {error ? <div className="auth-msg auth-msg--error">{error}</div> : null}
        {notice ? <div className="auth-msg auth-msg--notice">{notice}</div> : null}

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">
            Email
            <input
              className="auth-input"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </label>

          {mode !== "magic" ? (
            <label className="auth-label">
              Password
              <input
                className="auth-input"
                type="password"
                autoComplete={mode === "signup" ? "new-password" : "current-password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
              />
            </label>
          ) : null}

          <button className="auth-submit" type="submit" disabled={loading}>
            {loading ? "Please wait…" : mode === "login" ? "Sign in" : mode === "signup" ? "Create account" : "Send magic link"}
          </button>
        </form>
      </div>

      <style>{`
        .auth-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-sunken);
          font-family: var(--font-sans);
        }
        .auth-card {
          width: 100%;
          max-width: 380px;
          background: var(--bg-raised);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 32px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .auth-logo {
          display: flex;
          align-items: center;
          gap: 10px;
          color: var(--fg);
        }
        .auth-brand {
          font-size: 15px;
          font-weight: 600;
          letter-spacing: -0.01em;
          color: var(--fg);
        }
        .auth-brand-sub {
          color: var(--fg-3);
          font-weight: 400;
        }
        .auth-tabs {
          display: flex;
          gap: 2px;
          background: var(--bg-soft);
          border-radius: 8px;
          padding: 3px;
        }
        .auth-tab {
          flex: 1;
          padding: 6px 8px;
          border: none;
          border-radius: 6px;
          background: transparent;
          color: var(--fg-3);
          font-size: 12px;
          font-family: var(--font-sans);
          cursor: pointer;
          transition: background 0.15s, color 0.15s;
        }
        .auth-tab:hover { color: var(--fg-2); }
        .auth-tab--active {
          background: var(--bg-raised);
          color: var(--fg);
          font-weight: 500;
          box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .auth-msg {
          padding: 10px 12px;
          border-radius: 6px;
          font-size: 13px;
          line-height: 1.4;
        }
        .auth-msg--error {
          background: rgba(239, 68, 68, 0.12);
          color: var(--error-300);
          border: 1px solid rgba(239,68,68,0.25);
        }
        .auth-msg--notice {
          background: rgba(6, 215, 232, 0.08);
          color: var(--cyan-300);
          border: 1px solid rgba(6,215,232,0.2);
        }
        .auth-form {
          display: flex;
          flex-direction: column;
          gap: 14px;
        }
        .auth-label {
          display: flex;
          flex-direction: column;
          gap: 6px;
          font-size: 12px;
          font-weight: 500;
          color: var(--fg-3);
          letter-spacing: 0.02em;
          text-transform: uppercase;
        }
        .auth-input {
          padding: 9px 12px;
          border: 1px solid var(--border);
          border-radius: 7px;
          background: var(--bg-soft);
          color: var(--fg);
          font-size: 14px;
          font-family: var(--font-sans);
          outline: none;
          transition: border-color 0.15s;
        }
        .auth-input:focus {
          border-color: var(--indigo-500);
          box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
        }
        .auth-input::placeholder { color: var(--fg-4); }
        .auth-submit {
          margin-top: 4px;
          padding: 10px 16px;
          border: none;
          border-radius: 7px;
          background: var(--indigo-500);
          color: #fff;
          font-size: 14px;
          font-weight: 500;
          font-family: var(--font-sans);
          cursor: pointer;
          transition: background 0.15s, opacity 0.15s;
        }
        .auth-submit:hover:not(:disabled) { background: var(--indigo-400); }
        .auth-submit:disabled { opacity: 0.5; cursor: not-allowed; }
      `}</style>
    </div>
  );
}
