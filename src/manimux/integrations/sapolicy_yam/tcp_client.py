"""Minimal client for SAPolicy's length-prefixed JSON/numpy RPC service."""

from __future__ import annotations

import base64
import json
import socket
import threading
from collections.abc import Mapping
from contextlib import suppress

import numpy as np

_MAX_MESSAGE_BYTES = 256 << 20


class _NumpyEncoder(json.JSONEncoder):
    def default(self, value: object) -> object:
        if isinstance(value, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(value.tobytes()).decode("ascii"),
                "dtype": str(value.dtype),
                "shape": value.shape,
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return super().default(value)


def encode_message(value: object) -> bytes:
    return json.dumps(value, cls=_NumpyEncoder, separators=(",", ":")).encode("utf-8")


def decode_message(payload: bytes) -> object:
    def hook(value: dict[str, object]) -> object:
        if value.get("__numpy_array__") is True:
            data = value.get("data")
            dtype = value.get("dtype")
            shape = value.get("shape")
            if (
                not isinstance(data, str)
                or not isinstance(dtype, str)
                or not isinstance(shape, list)
            ):
                raise ValueError("invalid numpy value in SAPolicy response")
            raw = base64.b64decode(data, validate=True)
            parsed_dtype = np.dtype(dtype)
            if parsed_dtype.hasobject:
                raise ValueError("object arrays are forbidden in SAPolicy responses")
            parsed_shape = tuple(int(item) for item in shape)
            if any(item < 0 for item in parsed_shape):
                raise ValueError("negative numpy dimensions are forbidden")
            expected_bytes = int(np.prod(parsed_shape, dtype=np.int64)) * parsed_dtype.itemsize
            if expected_bytes != len(raw):
                raise ValueError("SAPolicy numpy payload byte length does not match its shape")
            return np.frombuffer(raw, dtype=parsed_dtype).reshape(parsed_shape)
        return value

    return json.loads(payload.decode("utf-8"), object_hook=hook)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("SAPolicy server closed the connection mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class SAPolicyTcpClient:
    """One serialized persistent connection; failed calls are never retried."""

    def __init__(self, host: str, port: int, *, connect_timeout_s: float) -> None:
        if not host:
            raise ValueError("SAPolicy host must be non-empty")
        if not 0 < port < 65536:
            raise ValueError("SAPolicy port must be between 1 and 65535")
        if connect_timeout_s <= 0:
            raise ValueError("SAPolicy connect timeout must be positive")
        self._host = host
        self._port = int(port)
        self._connect_timeout_s = float(connect_timeout_s)
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if self._socket is None:
            sock = socket.create_connection(
                (self._host, self._port), timeout=self._connect_timeout_s
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket = sock
        return self._socket

    def call(self, command: str, argument: object | None = None, *, timeout_s: float) -> object:
        if not command:
            raise ValueError("SAPolicy command must be non-empty")
        if timeout_s <= 0:
            raise TimeoutError("SAPolicy request deadline has elapsed")
        request = encode_message({"cmd": command, "obs": argument})
        if len(request) > _MAX_MESSAGE_BYTES:
            raise ValueError("SAPolicy request exceeds the wire size limit")

        with self._lock:
            sock = self._connect()
            sock.settimeout(timeout_s)
            try:
                sock.sendall(len(request).to_bytes(4, "big"))
                sock.sendall(request)
                size = int.from_bytes(_recv_exact(sock, 4), "big")
                if size <= 0 or size > _MAX_MESSAGE_BYTES:
                    raise ConnectionError(f"invalid SAPolicy response length {size}")
                response = decode_message(_recv_exact(sock, size))
            except BaseException:
                self._close_unlocked()
                raise

        if not isinstance(response, Mapping):
            raise ValueError("SAPolicy response must be a mapping")
        error = response.get("error")
        if error is not None:
            raise RuntimeError(f"SAPolicy server error: {error}")
        if "res" not in response:
            raise ValueError("SAPolicy response is missing 'res'")
        return response["res"]

    def _close_unlocked(self) -> None:
        if self._socket is None:
            return
        with suppress(OSError):
            self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()
        self._socket = None

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()
