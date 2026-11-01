#!/usr/bin/env python3
"""Create an auditable, causal release schedule for a 30 Hz reference NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from video2motion.timestamp_stream import make_timestamp_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument(
        "--frame-key",
        help="Array whose first dimension defines frame count (auto-detected if omitted)",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as reference:
        fps = float(reference["fps"])
        key = args.frame_key
        if key is None:
            key = next(
                (candidate for candidate in ("qpos", "body_quat", "smpl_joints") if candidate in reference),
                None,
            )
        if fps != 30.0 or key is None:
            raise ValueError("Expected a non-empty 30 Hz reference")
        frame_count = len(reference[key])
    signal = make_timestamp_index(frame_count, publish_hz=args.publish_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        signal_time=signal.signal_time,
        source_frame=signal.source_frame,
        source_time=signal.source_time,
        source_fps=np.array(fps),
        publish_hz=np.array(args.publish_hz),
    )


if __name__ == "__main__":
    main()
