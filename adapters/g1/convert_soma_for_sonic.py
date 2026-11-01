# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Convert current GEM-X native SOMA77 predictions to SONIC SMPL-mode input.

This is deliberately a narrow compatibility adapter.  SONIC's published
``soma_to_smpl.py`` targets an older SOMA runtime API; current GEM-X exposes
the fixed-GS1 SOMA77 adapter.  We evaluate that official adapter, select the
same 24 joints by their public SOMA names, and then apply SONIC's published
coordinate/quaternion convention without retargeting or amplitude scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sonic_bridge(sonic_root: Path):
    bridge_path = sonic_root / "gear_sonic/examples/live_camera_teleop/soma_to_smpl.py"
    spec = importlib.util.spec_from_file_location("sonic_soma_to_smpl", bridge_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SONIC bridge from {bridge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--gemx-root", required=True, type=Path)
    parser.add_argument("--sonic-root", required=True, type=Path)
    parser.add_argument("--soma-assets", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", default=30.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.gemx_root))
    sys.path.insert(0, str(args.sonic_root))

    from gem.common.soma.body import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
    from gem.motion.frontend.joint_sets import SOMA77_JOINT_NAMES
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion,
        quat_apply,
        quat_inv,
    )

    bridge = load_sonic_bridge(args.sonic_root)
    smpl_indices = bridge.build_soma_to_smpl_index(list(SOMA77_JOINT_NAMES))

    # Input is a trusted local artifact produced by GEM-X in this experiment.
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    params = payload.get("pred_soma_params_global")
    if not isinstance(params, dict):
        raise KeyError("GEM-X output lacks pred_soma_params_global")

    required = ("body_pose", "global_orient", "identity_coeffs", "scale_params", "transl")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"Missing SOMA parameters: {missing}")

    def tensor(key: str) -> torch.Tensor:
        value = torch.as_tensor(params[key], dtype=torch.float32, device=args.device)
        return value if value.ndim == 3 or key != "global_orient" else value

    body_pose = tensor("body_pose")
    global_orient = tensor("global_orient")
    identity = tensor("identity_coeffs")
    scale = tensor("scale_params")
    transl = tensor("transl")
    if body_pose.shape[0] != 1:
        raise ValueError(f"Expected one tracked person, got batch {body_pose.shape[0]}")

    body = SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(
            data_root=args.soma_assets,
            lod="low",
            mode="warp",
            device=args.device,
            apply_correctives=False,
        )
    ).to(args.device).eval()

    with torch.no_grad():
        decoded = body(
            global_orient=global_orient,
            body_pose=body_pose,
            identity_coeffs=identity,
            scale_params=scale,
            transl=transl,
            return_joints_only=True,
        )
        joints77 = decoded["joints"]
        if joints77 is None or joints77.shape[-2:] != (77, 3):
            raise ValueError(f"Unexpected SOMA joint output: {None if joints77 is None else joints77.shape}")
        index = torch.tensor(smpl_indices, dtype=torch.long, device=args.device)
        joints24 = joints77.index_select(-2, index)

        root_aa = global_orient.reshape(-1, 3)
        root_quat_y = angle_axis_to_quaternion(root_aa)
        body_quat = remove_smpl_base_rot(smpl_root_ytoz_up(root_quat_y), w_last=False)

        # SONIC live-camera convention: root-relative, Y-up -> Z-up, then root-local.
        yup_to_zup = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=args.device,
        )
        local_y = joints24 - joints24[..., 0:1, :]
        local_z = local_y @ yup_to_zup.T
        flat_joints = local_z.reshape(-1, 24, 3)
        inverse_root = quat_inv(body_quat).unsqueeze(1).expand(-1, 24, -1)
        smpl_joints = quat_apply(inverse_root.reshape(-1, 4), flat_joints.reshape(-1, 3))
        smpl_joints = smpl_joints.reshape(*joints24.shape[:-2], 24, 3)

    joints_np = smpl_joints.squeeze(0).cpu().numpy().astype(np.float32)
    quat_np = body_quat.reshape(1, -1, 4).squeeze(0).cpu().numpy().astype(np.float32)
    frames = int(joints_np.shape[0])
    wrists_np = np.zeros((frames, 6), dtype=np.float32)
    timestamps = (np.arange(frames, dtype=np.float64) / args.fps).astype(np.float64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        smpl_joints=joints_np,
        body_quat=quat_np,
        wrists=wrists_np,
        timestamps=timestamps,
        fps=np.array(args.fps, dtype=np.float32),
        soma_joint_names=np.asarray(SOMA77_JOINT_NAMES),
        smpl_joint_names=np.asarray(bridge.SMPL_24_NAMES),
        soma_to_smpl_indices=np.asarray(smpl_indices, dtype=np.int32),
        coordinate_convention=np.asarray("SONIC_SMPL_ROOT_LOCAL_Z_UP_WXYZ"),
    )

    metadata = {
        "format": "sonic_smpl_reference_v1",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "frames": frames,
        "fps": args.fps,
        "duration_seconds": frames / args.fps,
        "soma_public_joint_count": 77,
        "sonic_smpl_joint_count": 24,
        "soma_to_smpl_indices": smpl_indices,
        "soma_joint_names": list(SOMA77_JOINT_NAMES),
        "smpl_joint_names": list(bridge.SMPL_24_NAMES),
        "coordinate_convention": "SONIC SMPL root-local, Z-up, quaternion wxyz",
        "amplitude_scaling": "none",
        "temporal_smoothing": "none",
        "wrists": "six zeros; source video has no G1 wrist encoder channels",
        "compatibility_note": (
            "Uses current GEM-X SomaFixedGs1BodyAdapter and the public SOMA77 joint order, "
            "then SONIC's published joint aliases and coordinate transforms."
        ),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["output_sha256"] = sha256(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("frames", "fps", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
