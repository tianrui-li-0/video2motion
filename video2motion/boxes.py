"""Validate sparse, human-authored person boxes and interpolate per frame."""

from __future__ import annotations

import numpy as np


def interpolate_boxes(annotation, *, frames, width, height, video_sha256):
    if annotation["video_sha256"] != video_sha256:
        raise ValueError("Boxes belong to a different video")
    expected = (int(frames), int(width), int(height))
    actual = (annotation["frames"], annotation["width"], annotation["height"])
    if actual != expected:
        raise ValueError("Box timeline/dimensions differ from normalized video")

    rows = np.asarray(annotation["keyframes"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError("Expected finite [frame,x1,y1,x2,y2] keyframes")
    times, boxes = rows[:, 0], rows[:, 1:]
    if (
        times[0] != 0
        or times[-1] != frames - 1
        or not np.all(np.diff(times) > 0)
        or not np.all(times == times.astype(int))
    ):
        raise ValueError("Increasing integer keyframes must cover both endpoints")
    if (
        not np.all(boxes[:, 2:] > boxes[:, :2])
        or np.any(boxes < 0)
        or np.any(boxes[:, [0, 2]] > width)
        or np.any(boxes[:, [1, 3]] > height)
    ):
        raise ValueError("Boxes must have positive area and stay inside the video")

    timeline = np.arange(frames)
    return np.stack(
        [np.interp(timeline, times, boxes[:, axis]) for axis in range(4)], axis=1
    ).astype(np.float32)
