# Unitree G1 / SONIC adapter

This optional adapter demonstrates two consumers of the same extracted SOMA
motion:

1. `run_soma_retarget.py`: standard BVH → official SOMA Retargeter → 29-DoF G1
   kinematic reference, suitable for SONIC G1 mode.
2. `convert_soma_for_sonic.py`: decoded SOMA77 → SONIC's published 24-point
   root-local human-reference convention, suitable for SONIC human mode.

The first route places robot morphology and joint limits before policy tracking;
the second leaves human-to-robot adaptation to SONIC. They are intentionally not
frame-identical.

Upstream repositories and pinned commits are listed in the root README. No G1
mesh, SONIC policy or NVIDIA weight is copied into this repository.
