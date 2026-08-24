from __future__ import annotations

import json

import numpy as np

from manimux.recording.video import AsyncVideoRecorder
from manimux.types import SensorFrame


def _frame(timestamp_ns: int, value: int) -> SensorFrame:
    return SensorFrame(
        name="front camera",
        data=np.full((16, 16, 3), value, dtype=np.uint8),
        capture_monotonic_ns=timestamp_ns,
        sequence=value,
    )


def test_video_recording_can_be_disabled(tmp_path) -> None:
    recorder = AsyncVideoRecorder(tmp_path, fps=0, codec="mp4v", queue_size=2)

    recorder.submit({"front camera": _frame(0, 1)})
    summary = recorder.close()

    assert not summary.enabled
    assert not (tmp_path / "videos").exists()


def test_video_recording_writes_mp4_and_timestamp_index(tmp_path) -> None:
    recorder = AsyncVideoRecorder(tmp_path, fps=10, codec="mp4v", queue_size=4)
    recorder.submit({"front camera": _frame(0, 1)})
    recorder.submit({"front camera": _frame(50_000_000, 2)})
    recorder.submit({"front camera": _frame(100_000_000, 3)})

    summary = recorder.close()

    assert summary.enabled
    assert summary.error is None
    assert summary.frames_written == 2
    assert (tmp_path / "videos/front-camera.mp4").stat().st_size > 0
    index = json.loads((tmp_path / "videos/index.json").read_text(encoding="utf-8"))
    assert index["schema"] == "manimux-video-v1"
    assert index["cameras"]["front camera"]["capture_monotonic_ns"] == [0, 100_000_000]
