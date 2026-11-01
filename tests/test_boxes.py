import numpy as np
import pytest

from video2motion.boxes import interpolate_boxes


def test_interpolates_and_binds_boxes_to_video():
    annotation = {
        "video_sha256": "abc",
        "frames": 3,
        "width": 100,
        "height": 80,
        "keyframes": [[0, 10, 10, 30, 50], [2, 20, 12, 40, 52]],
    }
    boxes = interpolate_boxes(
        annotation, frames=3, width=100, height=80, video_sha256="abc"
    )
    assert boxes.dtype == np.float32
    np.testing.assert_allclose(boxes[1], [15, 11, 35, 51])
    with pytest.raises(ValueError, match="different video"):
        interpolate_boxes(
            annotation, frames=3, width=100, height=80, video_sha256="def"
        )
