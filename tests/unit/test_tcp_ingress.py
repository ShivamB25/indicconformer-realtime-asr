from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from scripts import tcp_ingress


class _Reader:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        wait_for: asyncio.Event | None = None,
    ) -> None:
        self._chunks: list[bytes] = chunks
        self._wait_for: asyncio.Event | None = wait_for

    async def read(self, _size: int = -1) -> bytes:
        if self._wait_for is not None:
            await self._wait_for.wait()
            self._wait_for = None
        return self._chunks.pop(0) if self._chunks else b""


class _Writer:
    def __init__(self, *, on_eof: Callable[[], None] | None = None) -> None:
        self.data = bytearray()
        self.closed: bool = False
        self._on_eof: Callable[[], None] | None = on_eof

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def can_write_eof(self) -> bool:
        return True

    def write_eof(self) -> None:
        if self._on_eof is not None:
            asyncio.get_running_loop().call_soon(self._on_eof)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_relay_drains_response_after_client_half_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_ready = asyncio.Event()
    server_reader = _Reader([b"response", b""], wait_for=response_ready)
    server_writer = _Writer(on_eof=response_ready.set)
    client_reader = _Reader([b"request", b""])
    client_writer = _Writer()

    async def open_server(
        _host: str,
        _port: int,
    ) -> tuple[tcp_ingress.Reader, tcp_ingress.Writer]:
        return server_reader, server_writer

    monkeypatch.setattr(asyncio, "open_connection", open_server)

    await tcp_ingress._forward(
        ("target", 8000),
        1.0,
        client_reader,
        client_writer,
    )

    assert server_writer.data == b"request"
    assert client_writer.data == b"response"
    assert server_writer.closed is True
    assert client_writer.closed is True
