from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "AIML Discovery"

# Legacy: kept so any existing imports don't break.
PROJECT_HOME = Path(
    os.getenv("AIML_DISCOVERY_HOME", Path.home() / ".aiml_discovery" / "projects")
).expanduser()

# Local file cache — used by both SQLite and Supabase backends to keep dataset
# and model files accessible on disk for pandas / joblib.  Already gitignored
# via the .aiml_discovery/ rule in .gitignore.
FILE_CACHE_DIR = Path(
    os.getenv("AIML_FILE_CACHE_DIR", Path.home() / ".aiml_discovery" / ".file_cache")
).expanduser()

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".db",
    ".sqlite",
    ".sqlite3",
}

UPLOAD_TYPES = ["csv", "xlsx", "xls", "json", "db", "sqlite", "sqlite3"]
