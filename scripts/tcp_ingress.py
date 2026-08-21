#!/usr/bin/env python3
"""Publish TCP listeners that relay to services on an isolated container network."""

from __future__ import annotations

import argparse
import asyncio
import math
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Protocol

_READ_BYTES = 64 * 1024
_DEFAULT_PEER_DRAIN_SECONDS = 30.0


class Reader(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


class Writer(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def can_write_eof(self) -> bool: ...

    def write_eof(self) -> None: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Route:
    listen_port: int
    target_host: str
    target_port: int


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _route(value: str) -> Route:
    parts = value.split(":")
    if len(parts) != 3 or not parts[1]:
        raise argparse.ArgumentTypeError("route must be LISTEN_PORT:TARGET_HOST:TARGET_PORT")
    return Route(_port(parts[0]), parts[1], _port(parts[2]))


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        action="append",
        required=True,
        type=_route,
        help="LISTEN_PORT:TARGET_HOST:TARGET_PORT; repeat for each listener",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="container address used for every listener",
    )
    parser.add_argument(
        "--peer-drain-seconds",
        default=_DEFAULT_PEER_DRAIN_SECONDS,
        type=_positive_seconds,
        help="maximum time to drain the opposite direction after EOF",
    )
    return parser


async def _pump(reader: Reader, writer: Writer) -> None:
    try:
        while data := await reader.read(_READ_BYTES):
            writer.write(data)
            await writer.drain()
    finally:
        if writer.can_write_eof():
            with suppress(ConnectionError, NotImplementedError, OSError):
                writer.write_eof()
                await writer.drain()


async def _close_writer(writer: Writer) -> None:
    writer.close()
    with suppress(ConnectionError, OSError):
        await writer.wait_closed()


async def _forward(
    target: tuple[str, int],
    peer_drain_seconds: float,
    client_reader: Reader,
    client_writer: Writer,
) -> None:
    try:
        server_reader, server_writer = await asyncio.open_connection(*target)
    except OSError:
        await _close_writer(client_writer)
        return

    tasks = {
        asyncio.create_task(_pump(client_reader, server_writer)),
        asyncio.create_task(_pump(server_reader, client_writer)),
    }
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        await asyncio.gather(*done, return_exceptions=True)
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=peer_drain_seconds,
                )
            except TimeoutError:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(_close_writer(server_writer), _close_writer(client_writer))


async def serve(
    routes: list[Route],
    *,
    listen_host: str,
    peer_drain_seconds: float,
) -> None:
    listen_ports = [route.listen_port for route in routes]
    if len(set(listen_ports)) != len(listen_ports):
        raise ValueError("route listen ports must be unique")

    servers = [
        await asyncio.start_server(
            partial(
                _forward,
                (route.target_host, route.target_port),
                peer_drain_seconds,
            ),
            listen_host,
            route.listen_port,
        )
        for route in routes
    ]
    try:
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers))


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    routes: list[Route] = arguments.route
    try:
        asyncio.run(
            serve(
                routes,
                listen_host=arguments.listen_host,
                peer_drain_seconds=arguments.peer_drain_seconds,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
