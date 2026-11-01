import numpy as np

from video2motion.timestamp_stream import make_timestamp_index


def test_timestamp_stream_never_reads_future_frames():
    signal = make_timestamp_index(60, lead_in_s=0, tail_s=0)
    assert len(signal.signal_time) == 100
    assert np.all(signal.source_time <= signal.signal_time + 1e-9)
    assert signal.source_frame[0] == 0
    assert signal.source_frame[-1] == 59
