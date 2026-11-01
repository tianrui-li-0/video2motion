# Pipeline boundaries

The core boundary is a robot-independent `MotionSequence`: world transforms in
meters, joint names, a topologically ordered parent array and FPS. BVH is the
interchange representation for tools that do not consume the NPZ directly.

## Stage A — owned pixels to native SOMA

Input is a video the operator has the right to process, plus sparse person
boxes authored from those pixels. The boxes are tied to the exact file hash.
The wrapper rejects detector/tracker configuration and blocks ONNX sessions
during GEM-X extraction, so the HumanArt YOLOX weight cannot be read by
accident.

The wrapper still calls upstream GEM-X for:

1. normalization and frame decoding;
2. native77 observations from the manual crops;
3. background camera-motion estimation;
4. body-pose feature extraction;
5. temporal SOMA motion prediction;
6. official contact translation and CCD IK post-processing.

Output is `pred_soma_params_global` in the GEM-X prediction archive.

## Optional adapter B1 — SOMA to G1 reference

`export_soma_bvh.py` decodes SOMA with the official body adapter, exports a BVH
from actual world transforms, and verifies a forward-kinematic round trip. The
official SOMA Retargeter then produces a 29-DoF G1 reference. SONIC runs in G1
mode and physically tracks that reference.

This route usually looks more robot-compatible because the kinematic target is
already joint-limited and expressed on G1 before policy tracking.

## Optional adapter B2 — SOMA directly to SONIC human mode

`convert_soma_for_sonic.py` selects SONIC's published 24 joint aliases from the
decoded SOMA77 body and applies SONIC's published Y-up-to-Z-up/root-local
convention. SONIC performs the human-to-robot adaptation inside the controller.

This route is shorter but its objective is a sparse human-body reference, not a
pre-retargeted G1 joint trajectory. The two outputs should not be expected to
match frame-for-frame.

## Stage C — controller and physics

Use the official SONIC C++ deployment path for deployment-like timing and the
official Unitree G1 model for physics. Cosmetic shells and props must stay
separate from the physical validation model. A pretty render is not sim-to-real
evidence.
