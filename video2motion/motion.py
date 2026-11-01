"""Portable world-transform motion artifact used between extractors and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionSequence:
    """An articulated motion independent of any destination robot.

    `world` has shape ``[frames, joints, 4, 4]`` and stores meters in a
    right-handed world frame. `parents` uses -1 for the single root.
    """

    world: np.ndarray
    joint_names: np.ndarray
    parents: np.ndarray
    fps: float

    def validate(self) -> "MotionSequence":
        world = np.asarray(self.world)
        names = np.asarray(self.joint_names)
        parents = np.asarray(self.parents)
        if world.ndim != 4 or world.shape[2:] != (4, 4) or len(world) < 1:
            raise ValueError("world must have shape [frames,joints,4,4]")
        joints = world.shape[1]
        if names.shape != (joints,) or parents.shape != (joints,):
            raise ValueError("joint names/parents must match the joint dimension")
        if np.count_nonzero(parents < 0) != 1:
            raise ValueError("motion must contain exactly one root")
        if np.any(parents >= np.arange(joints)):
            raise ValueError("parents must precede their children")
        if not np.isfinite(world).all() or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("motion and fps must be finite and fps positive")
        expected_bottom = np.broadcast_to([0.0, 0.0, 0.0, 1.0], world[..., 3, :].shape)
        if not np.allclose(world[..., 3, :], expected_bottom, atol=1e-6):
            raise ValueError("world entries must be homogeneous transforms")
        return self

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            world=np.asarray(self.world),
            joint_names=np.asarray(self.joint_names),
            parents=np.asarray(self.parents, dtype=np.int64),
            fps=np.asarray(float(self.fps)),
            units=np.asarray("meters"),
            transform_convention=np.asarray("world_from_joint"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MotionSequence":
        with np.load(path, allow_pickle=False) as data:
            if str(data["units"]) != "meters":
                raise ValueError("unsupported motion units")
            sequence = cls(
                world=data["world"],
                joint_names=data["joint_names"],
                parents=data["parents"],
                fps=float(data["fps"]),
            )
        return sequence.validate()
