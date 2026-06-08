from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .base import StorageBackend


@lru_cache(maxsize=1)
def get_backend() -> StorageBackend:
    """Return the active storage backend (SQLite locally, Supabase when env vars are set)."""
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"):
        from .supabase_backend import SupabaseBackend
        return SupabaseBackend(
            url=os.environ["SUPABASE_URL"],
            key=os.environ["SUPABASE_ANON_KEY"],
        )
    from .sqlite_backend import SQLiteBackend
    db_path = os.getenv("SQLITE_DB_PATH", str(Path.home() / ".aiml_discovery" / "automl.db"))
    return SQLiteBackend(db_path)


__all__ = ["get_backend", "StorageBackend"]
