"""Discover one physically stable LEAP grasp nominal for every HORA ball scale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.training import BackendAdapter, ensure_registries

SCALES = (0.8, 1.0, 1.2)
CACHE_PREFIX = "ball_grasp_allegro_new_physics_0731_50k"


def _scale_tag(scale: float) -> str:
    return f"{scale:g}"


def _compose_cfg(scale: float):
    with initialize_config_dir(
        config_dir=str((ROOT_DIR / "conf" / "ppo").resolve()),
        version_base="1.3",
    ):
        return compose(
            "config",
            overrides=[
                "task=leap_inhand_ball_grasp_allegro/mujoco",
                f"env.object_scale={scale:g}",
                "env.grasp_auto_save=false",
                "env.grasp_collection_target=1000000",
            ],
        )


def _env_override(scale: float) -> dict[str, Any]:
    cfg = _compose_cfg(scale)
    adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )
    return adapter.build_task_env_cfg_override()


def _cache_rows(scale: float) -> np.ndarray | None:
    cache = (
        ASSETS_ROOT_PATH
        / "robots"
        / "leap_hand"
        / "caches"
        / f"{CACHE_PREFIX}_{_scale_tag(scale)}.npy"
    )
    if not cache.exists():
        return None
    rows = np.asarray(np.load(cache), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 23:
        raise ValueError(f"Invalid existing scale cache {cache}: {rows.shape}")
    return rows


def _collect_candidates(scale: float, *, num_envs: int, target: int) -> np.ndarray:
    override = _env_override(scale)
    env = registry.make(
        "LeapInhandBallGraspAllegro",
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    actions = np.zeros((num_envs, 16), dtype=np.float32)
    max_steps = 50_000
    try:
        env.init_state()
        for step in range(1, max_steps + 1):
            env.step(actions)
            total = env._total_saved_grasps()
            if step % 100 == 0:
                print(f"scale={scale:g} step={step} candidates={total}/{target}", flush=True)
            if total >= target:
                return np.concatenate(env._saved_grasping_states, axis=0)[:target].astype(
                    np.float64
                )
    finally:
        env.close()
    raise RuntimeError(f"scale={scale:g} found fewer than {target} candidates")


def _representative_order(rows: np.ndarray) -> np.ndarray:
    aligned = np.asarray(rows, dtype=np.float64).copy()
    reference_quat = aligned[0, 19:23]
    flip = np.sum(aligned[:, 19:23] * reference_quat[None, :], axis=1) < 0.0
    aligned[flip, 19:23] *= -1.0
    center = np.median(aligned, axis=0)
    spread = np.median(np.abs(aligned - center), axis=0)
    spread = np.maximum(spread, np.asarray([0.02] * 16 + [0.001] * 3 + [0.01] * 4))
    score = np.sum(np.square((aligned - center) / spread), axis=1)
    return np.argsort(score)


def _find_verified_nominal(
    scale: float,
    rows: np.ndarray,
    *,
    num_envs: int = 64,
) -> tuple[np.ndarray, int, int]:
    override = _env_override(scale)
    override["grasp_seed_qpos"] = rows[0].tolist()
    override["domain_rand"]["joint_noise"] = 0.0
    env = registry.make(
        "LeapInhandBallGraspAllegro",
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    actions = np.zeros((num_envs, 16), dtype=np.float32)
    episode_steps = int(round(float(env.cfg.max_episode_seconds) / float(env.cfg.ctrl_dt)))
    try:
        for attempt, row_id in enumerate(_representative_order(rows), start=1):
            row = np.asarray(rows[int(row_id)], dtype=np.float64)
            env.cfg.grasp_seed_qpos = row.tolist()
            env._saved_grasping_states = []
            env._saved_grasp_keys = set()
            env._grasp_cache_saved = False
            env._last_grasp_auto_save_total = 0
            env._grasp_target_reached_notified = False
            env.init_state()
            for _ in range(episode_steps + 2):
                env.step(actions)
            verified = int(env._total_saved_grasps())
            if verified == num_envs:
                return row.copy(), verified, attempt
    finally:
        env.close()
    raise RuntimeError(
        f"scale={scale:g} has no deterministic nominal among {rows.shape[0]} candidates"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-target", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "logs" / "local_hora_pipeline" / "scale_nominals.json",
    )
    args = parser.parse_args()
    if args.candidate_target <= 0 or args.num_envs <= 0:
        raise ValueError("candidate-target and num-envs must be positive")

    ensure_registries()
    result: dict[str, Any] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scale in SCALES:
        rows = _cache_rows(scale)
        source = "existing_cache"
        if rows is None or rows.shape[0] < args.candidate_target:
            source = "collector_only"
            rows = _collect_candidates(
                scale,
                num_envs=args.num_envs,
                target=args.candidate_target,
            )
        candidate_rows = rows[: max(args.candidate_target, 1)]
        nominal, verified, attempts = _find_verified_nominal(scale, candidate_rows)
        result[_scale_tag(scale)] = {
            "source": source,
            "candidate_count": int(candidate_rows.shape[0]),
            "deterministic_settling_passed": verified,
            "verification_attempts": attempts,
            "pose": nominal.tolist(),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"scale={scale:g} nominal verified ({verified}/64)", flush=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
