"""Inspect random LEAP HORA cache rows in the native MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.manipulation.sharpa_inhand.base import resolve_grasp_cache_file

CACHE_PREFIX = "robots/leap_hand/caches/ball_grasp_hora_sharpa_style_50k"
SCALES = (0.8, 1.0, 1.2)
COLLECTOR_SIM_DT = 1.0 / 120.0


def _parse_scales(value: str) -> tuple[float, ...]:
    if value == "all":
        return SCALES
    scale = float(value)
    if scale not in SCALES:
        raise argparse.ArgumentTypeError(f"scale must be one of all, {SCALES}")
    return (scale,)


def _load_cache(scale: float, cache_prefix: str) -> tuple[Path, np.ndarray]:
    cache_file = resolve_grasp_cache_file(cache_prefix, scale)
    if not cache_file.is_absolute():
        cache_file = ASSETS_ROOT_PATH / cache_file
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing scale={scale:g} cache: {cache_file}")

    rows = np.load(cache_file, mmap_mode="r")
    if rows.ndim != 2 or rows.shape[1] != 23:
        raise ValueError(f"Expected cache shape (?, 23), got {rows.shape}: {cache_file}")
    if rows.shape[0] == 0:
        raise ValueError(f"Cache is empty: {cache_file}")
    if rows.dtype != np.float32:
        raise ValueError(f"Expected float32 cache, got {rows.dtype}: {cache_file}")
    if not np.isfinite(rows).all():
        raise ValueError(f"Cache contains non-finite values: {cache_file}")
    return cache_file, rows


def _build_scaled_model(scale: float) -> tuple[mujoco.MjModel, float]:
    scene = ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml"
    spec = mujoco.MjSpec.from_file(str(scene))

    visual_mesh = spec.mesh("leap_ball_visual_mesh")
    collision_geom = spec.geom("leap_object_col")
    rotation_marker = spec.geom("leap_ball_rotation_marker")
    scaled_radius = float(collision_geom.size[0]) * scale
    collision_geom.size[0] = scaled_radius
    visual_mesh.scale[:] = np.asarray(visual_mesh.scale) * scale
    rotation_marker.fromto[:] = np.asarray(rotation_marker.fromto) * scale
    rotation_marker.size[0] = float(rotation_marker.size[0]) * scale
    model = spec.compile()
    model.opt.timestep = COLLECTOR_SIM_DT
    return model, scaled_radius


def _show_cache_row(
    model: mujoco.MjModel,
    row: np.ndarray,
    *,
    title: str,
    frozen: bool,
    headless: bool,
) -> None:
    qpos = np.asarray(row, dtype=np.float64)
    if qpos.shape != (model.nq,):
        raise ValueError(f"Cache row has shape {qpos.shape}, model expects {(model.nq,)}")

    target = qpos[: model.nu].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = target
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos, qpos):
        raise RuntimeError("MuJoCo data did not retain the selected cache qpos")
    if not np.array_equal(data.ctrl, target):
        raise RuntimeError("MuJoCo data did not retain the selected cache control target")

    print(title)
    print(f"  hand qpos: {qpos[:16].tolist()}")
    print(f"  ball xyz+wxyz: {qpos[16:23].tolist()}")
    if headless:
        return

    if frozen:
        print("  Opening the exact pre-settling cache pose in the official MuJoCo Viewer.")
        print("  This frozen view intentionally does not advance physics.")
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)
        return

    if model.nkey < 1:
        raise RuntimeError("LEAP ball scene must provide keyframe 0 for cache playback")
    model.key_time[0] = 0.0
    model.key_qpos[0, :] = qpos
    model.key_qvel[0, :] = 0.0
    model.key_ctrl[0, :] = target

    def hold_cache_hand_target(_model: mujoco.MjModel, callback_data: mujoco.MjData) -> None:
        callback_data.ctrl[:] = target

    mujoco.set_mjcb_control(hold_cache_hand_target)
    try:
        print("  MuJoCo 3.8 native Viewer starts running immediately.")
        print("  Click Pause, set Key to 0, then click Load key to restore this exact cache row.")
        print("  After inspection, click Run to start physics settling.")
        print("  Close the window to continue to the next sampled cache row.")
        mujoco.viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)


def _sample_indices(row_count: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("--samples-per-scale must be positive")
    if sample_count > row_count:
        raise ValueError(
            f"Cannot sample {sample_count} unique rows from a cache containing {row_count} rows"
        )
    return np.asarray(rng.choice(row_count, size=sample_count, replace=False), dtype=np.int64)


def inspect_random_cache_rows(
    scales: Sequence[float],
    *,
    samples_per_scale: int,
    row_index: int | None,
    seed: int | None,
    cache_prefix: str,
    frozen: bool,
    headless: bool,
) -> None:
    rng = np.random.default_rng(seed)
    for scale in scales:
        cache_file, rows = _load_cache(scale, cache_prefix)
        if row_index is None:
            indices = _sample_indices(rows.shape[0], samples_per_scale, rng)
        else:
            if row_index < 0 or row_index >= rows.shape[0]:
                raise IndexError(
                    f"--row-index must be within 0..{rows.shape[0] - 1}, got {row_index}"
                )
            indices = np.asarray([row_index], dtype=np.int64)
        model, radius = _build_scaled_model(scale)
        print(
            f"\nscale={scale:g}: cache={cache_file}, rows={rows.shape[0]}, "
            f"ball radius={radius:.6f} m, sampled indices={indices.tolist()}"
        )
        for sample_number, sample_index in enumerate(indices, start=1):
            _show_cache_row(
                model,
                np.asarray(rows[sample_index]),
                title=(
                    f"[scale {scale:g}] sample {sample_number}/{samples_per_scale}, "
                    f"cache row {int(sample_index)}"
                ),
                frozen=frozen,
                headless=headless,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=_parse_scales,
        default=SCALES,
        metavar="{all,0.8,1.0,1.2}",
        help="Scale to inspect; defaults to all three scales.",
    )
    parser.add_argument(
        "--samples-per-scale",
        type=int,
        default=3,
        help="Number of unique random rows to inspect from each scale cache.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Inspect one exact cache row instead of sampling random rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible row selection.",
    )
    parser.add_argument("--cache-prefix", default=CACHE_PREFIX)
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Show the exact pre-settling row with fixed camera and no physics stepping.",
    )
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    inspect_random_cache_rows(
        args.scale,
        samples_per_scale=args.samples_per_scale,
        row_index=args.row_index,
        seed=args.seed,
        cache_prefix=args.cache_prefix,
        frozen=args.frozen,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
