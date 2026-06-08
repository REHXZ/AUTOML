const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function request(path, init) {
  const response = await fetch(apiUrl(path), {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (error) {
      throw new Error(
        "API returned a non-JSON response. Start the FastAPI backend on http://127.0.0.1:8000 and use the Vite dev server proxy."
      );
    }
  }

  if (!response.ok) {
    const detail = payload?.detail ?? response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return payload;
}

export function getHealth() {
  return request("/api/health");
}

export async function listProjects() {
  const payload = await request("/api/projects");
  return payload.projects;
}

export async function createProject(name, description) {
  const payload = await request("/api/projects", {
    method: "POST",
    body: JSON.stringify({ name, description })
  });
  return payload.project;
}

export async function listDatasets(projectId) {
  const payload = await request(
    `/api/projects/${encodeURIComponent(projectId)}/datasets`
  );
  return payload.datasets;
}

export async function uploadDataset(projectId, { file, name = "", tableName = "" }) {
  const form = new FormData();
  form.append("file", file);
  if (name.trim()) form.append("name", name.trim());
  if (tableName.trim()) form.append("table_name", tableName.trim());

  const response = await fetch(
    apiUrl(`/api/projects/${encodeURIComponent(projectId)}/datasets/upload`),
    {
      method: "POST",
      body: form
    }
  );
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(
        "API returned a non-JSON response. Start the FastAPI backend on http://127.0.0.1:8000 and use the Vite dev server proxy."
      );
    }
  }

  if (!response.ok) {
    const detail = payload?.detail ?? response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return payload.datasets;
}

export async function listSessions(projectId) {
  const payload = await request(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions`
  );
  return payload.sessions;
}

export function getSession(projectId, sessionId) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}`
  );
}

export function startSession(projectId, userGoal, apiKey) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions`,
    {
      method: "POST",
      body: JSON.stringify({ user_goal: userGoal, ...(apiKey ? { api_key: apiKey } : {}) })
    }
  );
}

export function deleteSession(projectId, sessionId) {
  return fetch(
    apiUrl(
      `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}`
    ),
    { method: "DELETE" }
  ).then(async (response) => {
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload?.detail ?? response.statusText);
    }
  });
}

export function submitAnswers(projectId, sessionId, answers) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}/answers`,
    {
      method: "POST",
      body: JSON.stringify({ answers })
    }
  );
}

export function sendFollowUp(projectId, sessionId, message, apiKey) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify({ message, ...(apiKey ? { api_key: apiKey } : {}) })
    }
  );
}

export function stopSession(projectId, sessionId) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}/stop`,
    { method: "POST" }
  );
}

export function notebookUrl(projectId, sessionId) {
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}/notebook`
  );
}

export function connectSessionEvents(projectId, sessionId, fromIndex, handlers) {
  const source = new EventSource(
    apiUrl(
      `/api/projects/${encodeURIComponent(projectId)}/autopilot/sessions/${encodeURIComponent(sessionId)}/events?from_index=${fromIndex}`
    )
  );

  source.addEventListener("step", (event) => {
    handlers.onStep(JSON.parse(event.data));
  });
  source.addEventListener("status", (event) => {
    handlers.onStatus(JSON.parse(event.data));
    source.close();
  });
  source.addEventListener("heartbeat", (event) => {
    handlers.onHeartbeat(JSON.parse(event.data));
  });
  source.onerror = () => {
    handlers.onError();
    source.close();
  };

  return () => source.close();
}

export function isTerminalStatus(status) {
  return status === "idle" || status === "complete" || status === "error";
}

export async function listRuns(projectId) {
  const payload = await request(
    `/api/projects/${encodeURIComponent(projectId)}/runs`
  );
  return payload.runs;
}

export function getRun(projectId, runId) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(runId)}`
  );
}

export function getRunCharts(projectId, runId) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(runId)}/charts`
  );
}

export async function scoreRunWithFile(projectId, runId, file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(
    apiUrl(`/api/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(runId)}/score`),
    { method: "POST", body: form }
  );
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error("API returned a non-JSON response.");
    }
  }
  if (!response.ok) {
    const detail = payload?.detail ?? response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export function scoreRunWithDataset(projectId, runId, datasetId) {
  return request(
    `/api/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(runId)}/score`,
    {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId })
    }
  );
}
