"""JWT authentication for the Flask API.

When SUPABASE_JWT_SECRET is set, every request must carry a valid Supabase-issued
JWT in the Authorization header (Bearer <token>) or in the `token` query parameter
(used by EventSource endpoints that cannot set custom headers).

When SUPABASE_JWT_SECRET is not set (local / offline dev), all requests are
accepted and the caller is assigned user_id "local".
"""
from __future__ import annotations

import os

from flask import request

_SECRET: str | None = os.getenv("SUPABASE_JWT_SECRET")


def get_current_user_id() -> str:
    """Return the authenticated user's UUID, or 'local' when auth is disabled."""
    if not _SECRET:
        return "local"

    token = _extract_token()
    if not token:
        from flask import abort
        abort(401, description="Authentication required. Include a Bearer token.")

    try:
        import jwt  # PyJWT
        payload = jwt.decode(
            token,
            _SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return str(payload["sub"])
    except Exception:
        from flask import abort
        abort(401, description="Invalid or expired token.")


def _extract_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip() or None
    return request.args.get("token") or None
