import numpy as np
import pytest

from video2motion.motion import MotionSequence


def test_motion_sequence_round_trip(tmp_path):
    world = np.broadcast_to(np.eye(4), (2, 2, 4, 4)).copy()
    world[:, 1, 0, 3] = [0.1, 0.2]
    sequence = MotionSequence(world, np.array(["root", "child"]), np.array([-1, 0]), 30)
    output = tmp_path / "motion.npz"
    sequence.save(output)
    loaded = MotionSequence.load(output)
    np.testing.assert_allclose(loaded.world, world)
    assert loaded.joint_names.tolist() == ["root", "child"]


def test_motion_sequence_rejects_non_topological_parents():
    world = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    with pytest.raises(ValueError, match="parents"):
        MotionSequence(world, np.array(["root", "child"]), np.array([-1, 1]), 30).validate()
