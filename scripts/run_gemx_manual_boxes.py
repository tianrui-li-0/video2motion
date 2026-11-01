# The condition assembly below adapts NVIDIA GEM-X gem/inference/body_video.py.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detector-free temporal GEM-X adapter, not a claim that motion-video has --boxes.

Only the source of the person boxes changes. Pose/ViTPose, camera estimation,
Motion and contact processing call the pinned upstream implementation.
All conditions are regenerated from pixels. No earlier detector caches are read.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from video2motion.hashes import sha256 as digest
from video2motion.boxes import interpolate_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gemx-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--soma-assets', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run = a.run.resolve()
    for directory in ('inputs', 'raw', 'processed', 'metrics'):
        (run / directory).mkdir(parents=True, exist_ok=True)
    output = run / 'raw/gemx_motion.pt'
    normalized = run / 'inputs/normalized.mp4'
    if output.exists() or normalized.exists():
        raise FileExistsError('Use a new run; retained attempts must not be overwritten')
    sys.path.insert(0, str(a.gemx_root.resolve()))
    import gem.inference.body_video as upstream
    from gem.inference.video import VideoFrames, normalize_video
    from gem.inference.postprocess import postprocess_motion_result
    from gem.inference.serialization import save_predictions, tensor_tree
    import onnxruntime as ort

    audit = dict(detector_constructor_attempts=0, onnx_session_attempts=0,
                 forbidden_asset_open_attempts=0, model_files_opened=[],
                 boxes_origin='Manually authored from input pixels; linear interpolation',
                 prior_detector_caches_used=False, seed=a.seed)
    opened = set()
    def no_detector(*args, **kwargs):
        audit['detector_constructor_attempts'] += 1
        raise RuntimeError('Automatic person detection is forbidden in this run')
    def no_onnx(*args, **kwargs):
        audit['onnx_session_attempts'] += 1
        raise RuntimeError('No ONNX inference is allowed in the GEM-X extraction process')
    def audit_open(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        name = str(args[0]).replace('\\', '/')
        lower = name.lower()
        if lower.endswith(('.onnx', '.ckpt', '.pth', '.pt', '.npz', '.pkl', '.usda')):
            if 'humanart' in lower or 'yolox' in lower or lower.endswith('.onnx'):
                audit['forbidden_asset_open_attempts'] += 1
                raise RuntimeError('Forbidden detector asset access: ' + name)
            opened.add(name)
    upstream.YOLOXONNX.__init__ = no_detector
    upstream.detect_track = no_detector
    ort.InferenceSession = no_onnx
    sys.addaudithook(audit_open)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    config = upstream.load_body_video_config(a.config)
    if 'detector' in config or 'tracking' in config:
        raise ValueError('Manual-only config must not contain an automatic detector or tracker')
    annotation = json.loads((run / 'inputs/manual-boxes.json').read_text(encoding='utf-8'))
    source = run / 'inputs/selected.mp4'
    source_sha = digest(source)
    print('Normalizing pixels; detector and ONNX constructors blocked', flush=True)
    metadata = normalize_video(source, normalized)
    try:
        with VideoFrames(normalized) as frames:
            boxes = interpolate_boxes(annotation, frames=len(frames), width=frames.width,
                                      height=frames.height, video_sha256=source_sha)
            # The official 3:4 crop formula, with its default 1.2 margin.
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            size = np.maximum(boxes[:, 3] - boxes[:, 1], (boxes[:, 2] - boxes[:, 0]) / .75) * 1.2
            xys = np.concatenate([center, size[:, None]], axis=1).astype(np.float32)
            np.savez_compressed(run / 'inputs/manual-boxes-per-frame.npz',
                boxes_xyxy=boxes, boxes_xys=xys, frame_ids=np.arange(len(frames)))
            expected = upstream.MotionConditionSpec(**config['motion']['condition_spec'])
            schema = upstream.get_observation_schema(expected.observation_joint_set)
            release = upstream.body_release(config)
            actual = upstream.MotionConditionSpec(f'{release.pose_release_id}:pose_tokens',
                config['observations']['release_id'], schema.name)
            if actual != expected:
                raise ValueError('Pose/observation releases differ from Motion training')
            camera_config = config.get('camera', {})
            intrinsics = upstream.motion_intrinsics(frames.width, frames.height, camera_config.get('intrinsics'))
            print('Extracting native77 observations from manual crops', flush=True)
            obs = config['observations']
            model = upstream.load_native77_detector(obs['checkpoint'], release_id=obs['release_id'],
                device=a.device, batch_size=obs.get('batch_size', 4))
            keypoints = model.predict(frames, xys)
            observation_release = model.release_id
            del model
            print('Estimating camera motion with manual foreground boxes', flush=True)
            rotations = camera_config.get('rotations')
            camera = upstream.camera_motion(frames, intrinsics.numpy(), boxes,
                mode=camera_config.get('mode', 'background'),
                rotations=np.load(rotations) if rotations is not None else None,
                keypoints=keypoints.numpy(), **camera_config.get('options', {}))
            print('Extracting frozen body Pose features', flush=True)
            model = upstream.load_pose_checkpoint(config['pose']['checkpoint'], release, device=a.device)
            features, feature_version = upstream.extract_body_features(frames, xys, intrinsics,
                model, batch_size=config['pose'].get('batch_size', 4), device=a.device)
            del model
            actual = upstream.MotionConditionSpec(feature_version, observation_release, schema.name)
            sequence = dict(condition_spec=asdict(actual), length=torch.tensor([len(frames)]),
                f_imgseq=features.unsqueeze(0), kp2d_native77=keypoints.unsqueeze(0),
                bbx_xys=torch.from_numpy(xys).unsqueeze(0),
                K_fullimg=intrinsics.expand(1, len(frames), 3, 3).clone(),
                cam_angvel=camera['cam_angvel'].unsqueeze(0),
                frame_times=(torch.arange(len(frames), dtype=torch.float64)/30).unsqueeze(0), sampling_fps=30.)
        print('Predicting continuous native SOMA motion', flush=True)
        predictor = upstream.load_predictor(config['motion']['checkpoint'], config['motion'],
            stats_path=config['motion'].get('stats'), device=a.device)
        result = upstream.run_motion(predictor, sequence, device=a.device)
        del predictor
        result = tensor_tree({**result, 'format': 'gem_body_video_predictions_v1',
            'video': metadata, 'release': config, 'conditions': sequence, 'camera': camera,
            'tracking': dict(boxes_xyxy=boxes, boxes_xys=xys, source='manual_keyframes')})
        print('Applying unchanged official contact translation and CCD IK', flush=True)
        result = postprocess_motion_result(result, soma_assets=str(a.soma_assets), device=a.device)
        save_predictions(result, output)
        audit.update(status='complete', frames=int(sequence['length'][0]),
                     source_sha256=source_sha, output_sha256=digest(output),
                     annotation_sha256=digest(run / 'inputs/manual-boxes.json'))
    finally:
        audit['model_files_opened'] = sorted(opened)
        audit['scope'] = 'Extraction process only; downstream SONIC intentionally uses its own ONNX policies'
        (run / 'metrics/detector-bypass.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
