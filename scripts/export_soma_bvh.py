# SPDX-License-Identifier: Apache-2.0
"""Decode saved native GEM-X SOMA motion and export its actual FK to SOMA BVH.

Never interpret SOMA pose parameters as BVH Euler angles: bind orientations
are part of the SOMA model. Use decoded world transforms, including Root.
Six channels per joint preserve the fitted skeleton, without an SMPL backend.
"""
import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.spatial.transform import Rotation
from video2motion.motion import MotionSequence


def write_bvh(path, names, parents, world, fps):
    local = world.copy().astype(np.float64)
    for j, parent in enumerate(parents):
        if parent >= 0:
            local[:, j] = np.linalg.inv(world[:, parent].astype(np.float64)) @ world[:, j]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # Euler gimbal lock: roundtrip checked below.
        angles = Rotation.from_matrix(local[..., :3, :3].reshape(-1, 3, 3)).as_euler('XYZ', degrees=True)
    angles = angles.reshape(len(world), len(names), 3)
    reconstructed_local = local.copy()
    reconstructed_local[..., :3, :3] = Rotation.from_euler('XYZ', angles.reshape(-1, 3), degrees=True).as_matrix().reshape(local[..., :3, :3].shape)
    reconstructed = reconstructed_local.copy()
    order = []
    lines = ['HIERARCHY']

    def joint(j, depth):
        order.append(j)
        pad = '  ' * depth
        lines.extend([f'{pad}{"ROOT" if parents[j] < 0 else "JOINT"} {names[j]}', pad + '{',
                      pad + '  OFFSET ' + ' '.join(f'{v:.9f}' for v in local[0, j, :3, 3] * 100),
                      pad + '  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation'])
        children = np.flatnonzero(parents == j)
        for child in children:
            joint(int(child), depth + 1)
        if not len(children):
            lines.extend([pad + '  End Site', pad + '  {', pad + '    OFFSET 0 0 0', pad + '  }'])
        lines.append(pad + '}')

    roots = np.flatnonzero(parents < 0)
    assert len(roots) == 1
    joint(int(roots[0]), 0)
    for j in order:
        if parents[j] >= 0:
            reconstructed[:, j] = reconstructed[:, parents[j]] @ reconstructed_local[:, j]
    error = float(np.max(np.abs(reconstructed - world)))
    assert error < 2e-5, error
    lines.extend(['MOTION', f'Frames: {len(world)}', f'Frame Time: {1 / fps:.12f}'])
    values = np.concatenate([np.take(local, order, axis=1)[..., :3, 3] * 100, angles[:, order]], axis=-1)
    lines.extend(' '.join(f'{v:.9f}' for v in frame.ravel()) for frame in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gemx-root', type=Path, required=True)
    p.add_argument('--soma-assets', type=Path, required=True)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--reuse-decoded', action='store_true')
    a = p.parse_args()
    for directory in ('processed', 'metrics'):
        (a.run / directory).mkdir(parents=True, exist_ok=True)
    if a.reuse_decoded:
        native = np.load(a.run / 'processed/soma_world_transforms.npz')
        finish(a, native['joint_names'], native['parents'], native['world'])
        return
    import torch
    sys.path.insert(0, str(a.gemx_root))
    from gem.common.soma.body import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
    params = torch.load(a.input, map_location='cpu', weights_only=False)['pred_soma_params_global']
    keys = ['body_pose', 'global_orient', 'identity_coeffs', 'scale_params', 'transl']
    params = {k: torch.as_tensor(params[k], dtype=torch.float32, device='cuda') for k in keys}
    body = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=a.soma_assets, lod='low',
        mode='warp', device='cuda', apply_correctives=False)).to('cuda').eval()
    with torch.no_grad():
        decoded = body(**params, return_joints_only=True)
    world = decoded['transforms'].squeeze(0).cpu().numpy()
    joints = decoded['joints'].squeeze(0).cpu().numpy()
    names = np.array(body.runtime.public_joint_names)
    parents = body.runtime.public_joint_parent_ids.cpu().numpy().astype(np.int64)
    # SOMA represents the virtual Root as self-parented; BVH uses -1.
    assert parents[0] == 0 and np.all(parents[1:] < np.arange(1, len(parents)))
    parents[0] = -1
    assert world.shape[1:] == (78, 4, 4) and names[0] == 'Root'
    assert np.allclose(world[:, 1:, :3, 3], joints, atol=1e-6)
    native_path = a.run / 'processed/soma_world_transforms.npz'
    if native_path.exists():
        raise FileExistsError('Preserve existing decode; use --reuse-decoded or a new run')
    MotionSequence(world, names, parents, a.fps).save(native_path)
    finish(a, names, parents, world)


def finish(a, names, parents, world):
    error = write_bvh(a.run / 'processed/soma_bvh/selected.bvh', names, parents, world, a.fps)
    report = dict(frames=len(world), fps=a.fps, joints=len(names),
        native_to_bvh_fk_max_abs_error=error, units='BVH cm; native NPZ meters',
        rotations='intrinsic XYZ degrees from actual SOMA world FK',
        source=str(a.input), smoothing='none', rescaling='none',
        note='SOMA-X native body only; no SMPL/SMPL-X model or parameters used')
    (a.run / 'metrics/soma-bvh-export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
