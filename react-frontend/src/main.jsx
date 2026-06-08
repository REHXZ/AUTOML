import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import AuthPage from "./components/AuthPage";
import LandingPage from "./components/LandingPage";
import { setAuthToken } from "./api";
import { supabase } from "./lib/supabase";
import "./styles/index.css";

function Root() {
  // "checking" | "auth" | "landing" | "app"
  const [view, setView] = useState(supabase ? "checking" : "landing");
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (!supabase) return;

    supabase.auth.getSession().then(({ data: { session } }) => {
      const token = session?.access_token ?? null;
      setAuthToken(token);
      setUser(session?.user ?? null);
      setView(session ? "landing" : "auth");
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      const token = session?.access_token ?? null;
      setAuthToken(token);
      setUser(session?.user ?? null);
      if (!session) {
        setView("auth");
      } else if (event === "SIGNED_IN") {
        // OAuth / magic-link just completed — advance past the auth page
        setView("landing");
      }
      // TOKEN_REFRESHED / USER_UPDATED don't change the view
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    if (!supabase) {
      // Auth is disabled (no env vars) — just return to landing
      setView("landing");
      return;
    }
    await supabase.auth.signOut();
    setAuthToken(null);
    setUser(null);
    setView("auth");
  };

  if (view === "checking") {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg-sunken)",
        color: "var(--fg-3)",
        fontFamily: "var(--font-sans)",
        fontSize: "13px"
      }}>
        Loading…
      </div>
    );
  }

  if (view === "auth") {
    return <AuthPage onSuccess={() => setView("landing")} />;
  }

  if (view === "app") {
    return <App user={user} onSignOut={supabase ? handleSignOut : undefined} />;
  }

  return (
    <LandingPage
      onEnterApp={() => {
        // Gate entry behind auth when Supabase is configured and user isn't signed in
        if (supabase && !user) setView("auth");
        else setView("app");
      }}
      user={user}
      onSignOut={supabase ? handleSignOut : undefined}
    />
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
