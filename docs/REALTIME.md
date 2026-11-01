# What “real-time” means here

The repository separates three different clocks:

1. **Video extraction** — currently offline GEM-X processing.
2. **Reference release** — a prerecorded 30 Hz motion is revealed by timestamp.
3. **Control** — official SONIC consumes the available reference at 50 Hz.

`make_timestamp_index` implements stage 2 with `floor(t * source_fps)`. It is a
useful causal controller test: at time `t`, the publisher cannot access a source
frame later than `t`. It is not evidence that stage 1 can execute online.

For actual teleoperation, replace stage 1 with a tracker/VR/exoskeleton stream
that already emits current body targets. Keep the `MotionSequence`/adapter
boundary and measure sensor, transport, retargeting, policy, actuator and
mechanical delays end-to-end on hardware.
