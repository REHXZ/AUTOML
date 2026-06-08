import React, { useState } from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import LandingPage from "./components/LandingPage";
import "./styles/index.css";

function Root() {
  const [view, setView] = useState("landing");

  if (view === "app") return <App />;
  return <LandingPage onEnterApp={() => setView("app")} />;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
