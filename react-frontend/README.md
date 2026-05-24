# React Autopilot Dashboard

React JavaScript dashboard for the AIML Discovery AI Autopilot.

## Run Locally

Start the FastAPI backend from the repository root:

```powershell
uvicorn aiml_discovery.api:app --reload
```

Start the React frontend:

```powershell
cd react-frontend
npm.cmd install
npm.cmd run dev
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173
```

The dev server proxies `/api` to `http://127.0.0.1:8000`, so CORS is not needed for local development.

## Notes

- The dashboard uses projects and datasets already registered through the existing app/API.
- Browser uploads are not part of this v1.
- The frontend does not collect API keys. Set `OPENAI_API_KEY` in the backend environment or `.env`.
- To point the frontend at another backend, set `VITE_API_BASE` before running Vite.
