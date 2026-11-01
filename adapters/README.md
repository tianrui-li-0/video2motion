# Adapters

An adapter consumes the portable `MotionSequence` NPZ or exported BVH and emits
the input contract of a destination skeleton, animation package or controller.

Adapters must not modify extraction, silently rescale time/amplitude, or hide
joint-limit clamping. They should retain input/output hashes and report every
coordinate-system conversion. The bundled `g1` adapter is the reference example.
