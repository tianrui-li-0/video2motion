#!/usr/bin/env python3
"""Download the minimal official NVIDIA inference files at fixed revisions.

The repository intentionally contains no model binaries. This script excludes
the HumanArt detector and all training datasets/SMPL archives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from video2motion.hashes import sha256


MODELS = {
    "gemx": {
        "repo_id": "nvidia/GEM-X",
        "revision": "4fce057963671fd95cd4154297944397e7b5a9a0",
        "files": [
            "LICENSE",
            "body_config.json",
            "body_s240000.ckpt",
            "body_video_native77.yaml",
            "motion_native77.yaml",
            "motion_native77_e774_s292175.ckpt",
            "motion_stats_raw.npz",
            "vitpose.pth",
        ],
        "sha256": {
            "body_s240000.ckpt": "e244d27f2834eb6382c5f6b9cbf2a38681b685514a1b2e86f8f3d2592d614959",
            "motion_native77_e774_s292175.ckpt": "c7790f4587e518bcea40cb8281c9be2233d37f6caa668835462542760dcf82aa",
            "motion_stats_raw.npz": "b7719b634e0fc1f001005605d117d77448c8d4a110923149df58529c68112ed6",
            "vitpose.pth": "8e3eec71dbd6bb745eba8c6debe3e4b97bcfd1d01f300099f2eedf50b9a80c8e",
        },
    },
    "sonic": {
        "repo_id": "nvidia/GEAR-SONIC",
        "revision": "6733128a3d8a523b1418b06bca3cdf61c8b0987f",
        "files": ["LICENSE", "config.json", "low_latency/*"],
        "sha256": {
            "low_latency/model_encoder.onnx": "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b",
            "low_latency/model_decoder.onnx": "c4ac2e74045e7cbfb568f15e6bf47ea7ce023df7a94322af50be223e0a628bab",
            "low_latency/observation_config.yaml": "582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab",
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("weights"))
    parser.add_argument("--only", choices=["gemx", "sonic"], action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.only or ["gemx"]
    plan = {name: MODELS[name] for name in selected}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    manifest = {}
    for name in selected:
        spec = MODELS[name]
        destination = (args.output / name).resolve()
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=spec["files"],
            local_dir=destination,
        )
        verified = {}
        for relative, expected in spec["sha256"].items():
            path = destination / relative
            actual = sha256(path)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")
            verified[relative] = actual
        manifest[name] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "files": verified,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
