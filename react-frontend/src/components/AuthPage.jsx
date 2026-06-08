import { useState } from "react";
import { supabase } from "../lib/supabase";

const SSO_PROVIDERS = [
  {
    id: "google",
    label: "Continue with Google",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
      </svg>
    ),
  },
  {
    id: "github",
    label: "Continue with GitHub",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
      </svg>
    ),
  },
];

export default function AuthPage({ onSuccess }) {
  const [mode, setMode] = useState("login"); // login | signup | magic
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [ssoLoading, setSsoLoading] = useState(null);
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

  const handleSSO = async (provider) => {
    if (!supabase) {
      setError("Authentication is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.");
      return;
    }
    reset();
    setSsoLoading(provider);
    try {
      const { error: err } = await supabase.auth.signInWithOAuth({
        provider,
        options: { redirectTo: window.location.origin },
      });
      if (err) throw err;
      // Page will redirect — no further action needed here
    } catch (err) {
      setError(err.message ?? `${provider} sign-in failed.`);
      setSsoLoading(null);
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

        {/* SSO buttons */}
        <div className="auth-sso">
          {SSO_PROVIDERS.map(({ id, label, icon }) => (
            <button
              key={id}
              type="button"
              className="auth-sso-btn"
              disabled={!!ssoLoading || loading}
              onClick={() => handleSSO(id)}
            >
              {ssoLoading === id ? <span className="auth-sso-spinner" /> : icon}
              {label}
            </button>
          ))}
        </div>

        <div className="auth-divider"><span>or</span></div>

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

          <button className="auth-submit" type="submit" disabled={loading || !!ssoLoading}>
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
        .auth-sso {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .auth-sso-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          padding: 10px 16px;
          border: 1px solid var(--border);
          border-radius: 7px;
          background: var(--bg-soft);
          color: var(--fg-2);
          font-size: 13px;
          font-weight: 500;
          font-family: var(--font-sans);
          cursor: pointer;
          transition: background 0.15s, border-color 0.15s, color 0.15s;
        }
        .auth-sso-btn:hover:not(:disabled) {
          background: var(--bg-raised);
          border-color: var(--fg-4);
          color: var(--fg);
        }
        .auth-sso-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .auth-sso-spinner {
          width: 14px;
          height: 14px;
          border: 2px solid var(--fg-4);
          border-top-color: var(--fg);
          border-radius: 50%;
          animation: auth-spin 0.6s linear infinite;
          flex-shrink: 0;
        }
        @keyframes auth-spin { to { transform: rotate(360deg); } }
        .auth-divider {
          display: flex;
          align-items: center;
          gap: 12px;
          color: var(--fg-4);
          font-size: 12px;
        }
        .auth-divider::before,
        .auth-divider::after {
          content: '';
          flex: 1;
          height: 1px;
          background: var(--border);
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
