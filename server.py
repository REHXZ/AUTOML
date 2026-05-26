"""Entry point for the AIML Discovery Flask API.

Usage:
    python server.py

Environment variables:
    API_HOST  — bind address (default: 0.0.0.0)
    API_PORT  — port (default: 8082)
"""

from aiml_discovery.api import HOST, PORT, app

if __name__ == "__main__":
    print(f"Starting AIML Discovery API on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
