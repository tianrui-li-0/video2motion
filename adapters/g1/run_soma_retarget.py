# SPDX-License-Identifier: Apache-2.0
"""Run official SOMA->G1 Newton pipeline on decoded GEM-X BVH; retain CSV."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import newton
import warp as wp
from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.io.csv import save_csv
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.utils.newton_utils import get_filtered_joint_names
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--soma-retargeter-root', type=Path, required=True)
    a = p.parse_args()
    for directory in ('raw', 'metrics'):
        (a.run / directory).mkdir(parents=True, exist_ok=True)
    out = a.run / 'raw' / datetime.now(timezone.utc).strftime('soma_retarget_%Y%m%dT%H%M%S%fZ')
    out.mkdir()
    shutil.copy2(__file__, out / 'runner.py')
    wp.init()
    skeleton, animation = load_bvh(str(a.run / 'processed/soma_bvh/selected.bvh'))
    native = np.load(a.run / 'processed/soma_world_transforms.npz', allow_pickle=False)
    local = np.asarray(animation.local_transforms)
    world = np.broadcast_to(np.eye(4), (*local.shape[:2], 4, 4)).copy()
    world[..., :3, :3] = Rotation.from_quat(local[..., 3:7].reshape(-1, 4)).as_matrix().reshape(*local.shape[:2], 3, 3)
    world[..., :3, 3] = local[..., :3]
    for j, parent in enumerate(skeleton.parent_indices):
        if parent >= 0:
            world[:, j] = world[:, parent] @ world[:, j]
    order = [list(native['joint_names']).index(n) for n in skeleton.joint_names]
    truth = native['world'][:, order]
    position_error = float(np.max(np.linalg.norm(world[..., :3, 3] - truth[..., :3, 3], axis=-1)))
    rotation_error = float(np.degrees(Rotation.from_matrix(
        (np.swapaxes(world[..., :3, :3], -1, -2) @ truth[..., :3, :3]).reshape(-1, 3, 3)).magnitude()).max())
    assert position_error < 1e-4 and rotation_error < .01, (position_error, rotation_error)
    np.savez_compressed(out / 'bvh_roundtrip.npz', world=world, joint_names=skeleton.joint_names)
    pipeline = SomaRetargetingPipeline(skeleton, 'soma', 'unitree_g1')
    # Exactly the official batch app's Y-up source -> Newton Z-up conversion.
    converter = SpaceConverter(FacingDirectionType.MUJOCO)
    pipeline.add_input_motions([animation], [converter.transform(wp.transform_identity())], True)
    result = pipeline.execute()[0]
    names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    assert result.num_frames == len(native['world']) and abs(result.sample_rate - float(native['fps'])) < 1e-5
    save_csv(str(out / 'selected.csv'), names, result)
    configs = a.soma_retargeter_root / 'soma_retargeter/assets/robotics/unitree/g1/configs'
    shutil.copytree(configs, out / 'official-configs')
    metadata = dict(status='retarget_complete', raw_directory=str(out), csv=str(out / 'selected.csv'),
        frames=result.num_frames, fps=result.sample_rate, joint_names=names,
        bvh_roundtrip_position_error_m=position_error, bvh_roundtrip_rotation_error_deg=rotation_error,
        settings='Official G1 defaults: scaling 0.78514868, IK24, contact/postprocessing enabled, joint clamps',
        source_facing='Mujoco: rotate Y-up source +90deg around X',
        note='Kinematic reference; not a physics rollout. No user-video ownership assertion.')
    (out / 'result.json').write_text(json.dumps(metadata, indent=2))
    (a.run / 'metrics/soma-retarget.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
