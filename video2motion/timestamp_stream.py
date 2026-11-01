"""Turn an offline 30 Hz reference into causal, timestamp-gated 50 Hz packets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimestampedSignal:
    signal_time: np.ndarray
    source_frame: np.ndarray
    source_time: np.ndarray


def make_timestamp_index(
    frame_count: int,
    *,
    source_fps: float = 30.0,
    publish_hz: float = 50.0,
    lead_in_s: float = 3.0,
    tail_s: float = 2.0,
) -> TimestampedSignal:
    """Create a zero-lookahead release schedule.

    Each packet uses floor(current_time * source_fps), never a future source
    frame. Lead-in and tail packets hold the first/last reference frame.
    """
    if frame_count < 1 or source_fps <= 0 or publish_hz <= 0:
        raise ValueError("frame_count and frequencies must be positive")
    duration = frame_count / source_fps
    count = int(np.ceil((lead_in_s + duration + tail_s) * publish_hz))
    signal_time = np.arange(count, dtype=np.float64) / publish_hz - lead_in_s
    source_frame = np.floor(signal_time * source_fps + 1e-9).astype(np.int64)
    source_frame = np.clip(source_frame, 0, frame_count - 1)
    source_time = source_frame.astype(np.float64) / source_fps
    if np.any(source_time[signal_time >= 0] - signal_time[signal_time >= 0] > 1e-9):
        raise AssertionError("Schedule released a future frame")
    return TimestampedSignal(signal_time, source_frame, source_time)
