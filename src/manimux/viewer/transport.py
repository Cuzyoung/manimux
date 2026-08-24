"""Small localhost bridge shared by arbitrary robot and policy integrations."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any, cast

import zmq


class ViewerPublisher:
    """Non-blocking publisher used by the robot/policy process."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5568") -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._socket: zmq.Socket[bytes] = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def publish(self, message: Any) -> None:
        payload = message.to_wire() if hasattr(message, "to_wire") else message
        with contextlib.suppress(zmq.Again):
            self._socket.send_json(payload, flags=zmq.NOBLOCK)

    def close(self) -> None:
        self._socket.close(linger=0)


class ViewerReceiver:
    def __init__(self, endpoint: str, callback: Callable[[dict[str, Any]], None]) -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._socket: zmq.Socket[bytes] = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, 4)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._run, name="viewer-bridge", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            if self._socket in dict(poller.poll(100)):
                self._callback(cast(dict[str, Any], self._socket.recv_json()))

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._socket.close(linger=0)


class ControlServer:
    """Serve dashboard controls to one or more policy executors."""

    def __init__(self, endpoint: str, callback: Callable[[], dict[str, Any]]) -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._socket: zmq.Socket[bytes] = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._run, name="viewer-controls", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            if self._socket in dict(poller.poll(100)):
                self._socket.recv()
                self._socket.send_json(self._callback())

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._socket.close(linger=0)


class ControlClient:
    """Best-effort polling client that fails closed when the viewer is absent."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5569") -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._endpoint = endpoint
        self._socket: zmq.Socket[bytes] | None = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

    def poll(self) -> dict[str, Any]:
        assert self._socket is not None
        try:
            self._socket.send(b"state", flags=zmq.NOBLOCK)
            if self._socket.poll(20, zmq.POLLIN):
                return cast(dict[str, Any], self._socket.recv_json())
        except zmq.ZMQError:
            pass
        self._connect()
        return {
            "paused": True,
            "home_requested": False,
            "finish_requested": False,
        }

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
