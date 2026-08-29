"""Shared bearer authentication for every inference transport."""

import secrets

from fastapi import HTTPException, Request, WebSocket, status


def _api_key_configured(connection: Request | WebSocket) -> tuple[bool, str | None]:
    expected = getattr(connection.app.state, "api_key", None)
    if isinstance(expected, str):
        return True, expected

    settings = getattr(connection.app.state, "settings", None)
    configured = settings is not None and settings.api_key_file is not None
    return configured, None


def _valid_bearer(connection: Request | WebSocket, expected: str) -> bool:
    values = connection.headers.getlist("authorization")
    if len(values) != 1:
        return False
    parts = values[0].split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        return False
    return secrets.compare_digest(parts[1].encode("utf-8"), expected.encode("ascii"))


def require_http_api_key(request: Request) -> None:
    """Require the configured service API key, while keeping keyless local use open."""
    configured, expected = _api_key_configured(request)
    if not configured:
        return
    if expected is None or not _valid_bearer(request, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def websocket_admitted(websocket: WebSocket) -> bool:
    """Check exact Origin policy and the shared bearer before WebSocket acceptance."""
    settings = getattr(websocket.app.state, "settings", None)
    origin = websocket.headers.get("origin")
    if (
        settings is not None
        and origin is not None
        and origin not in settings.websocket_allowed_origins
    ):
        return False

    configured, expected = _api_key_configured(websocket)
    if not configured:
        return True
    return expected is not None and _valid_bearer(websocket, expected)
