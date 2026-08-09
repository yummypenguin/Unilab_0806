"""Display the LEAP HORA grasp-generation nominal pose in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.assets import ASSETS_ROOT_PATH

NOMINAL_CONFIG = (
    ROOT_DIR / "conf" / "ppo" / "task" / "leap_inhand_ball_grasp_allegro" / "mujoco.yaml"
)


def _load_scale_nominal(scale: float) -> np.ndarray:
    del scale
    config = OmegaConf.load(NOMINAL_CONFIG)
    nominal_qpos = np.asarray(config.env.grasp_seed_qpos, dtype=np.float64)
    if nominal_qpos.shape != (23,) or not np.isfinite(nominal_qpos).all():
        raise ValueError(f"Invalid shared nominal pose: {nominal_qpos.shape}")
    return nominal_qpos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=float,
        choices=(0.8, 1.0, 1.2),
        default=0.8,
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.5,
        help="Simulate this many seconds before opening the frozen viewer.",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=1.0 / 120.0,
        help="Physics timestep; defaults to the LEAP grasp collector owner config.",
    )
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.settle_seconds < 0.0:
        raise ValueError("--settle-seconds must be non-negative")
    if args.sim_dt <= 0.0:
        raise ValueError("--sim-dt must be positive")

    scene = ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml"
    spec = mujoco.MjSpec.from_file(str(scene))

    visual_mesh = spec.mesh("leap_ball_visual_mesh")
    collision_geom = spec.geom("leap_object_col")
    rotation_marker = spec.geom("leap_ball_rotation_marker")
    base_radius = float(collision_geom.size[0])
    scaled_radius = base_radius * float(args.scale)
    collision_geom.size[0] = scaled_radius
    visual_mesh.scale[:] = np.asarray(visual_mesh.scale) * float(args.scale)
    rotation_marker.fromto[:] = np.asarray(rotation_marker.fromto) * float(args.scale)
    rotation_marker.size[0] = float(rotation_marker.size[0]) * float(args.scale)

    model = spec.compile()
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)

    nominal_qpos = _load_scale_nominal(args.scale)
    if nominal_qpos.shape != (model.nq,):
        raise ValueError(f"Nominal qpos has {nominal_qpos.shape}, model expects {(model.nq,)}")
    data.qpos[:] = nominal_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = nominal_qpos[: model.nu]
    mujoco.mj_forward(model, data)

    initial_ball_pose = data.qpos[16:23].copy()
    settle_steps = int(round(float(args.settle_seconds) / float(model.opt.timestep)))
    for _ in range(settle_steps):
        # Hold the nominal hand joint target while the ball and contacts settle.
        data.ctrl[:] = nominal_qpos[: model.nu]
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    settled_ball_pose = data.qpos[16:23].copy()

    print(f"LEAP HORA nominal pose: scale={args.scale:g}, ball radius={scaled_radius:.6f} m")
    print(
        f"Settled for {settle_steps * model.opt.timestep:.3f} s "
        f"({settle_steps} physics steps, dt={model.opt.timestep:g} s)"
    )
    print(f"Initial ball pose xyz+wxyz: {initial_ball_pose.tolist()}")
    print(f"Settled ball pose xyz+wxyz: {settled_ball_pose.tolist()}")
    if args.headless:
        return
    print("Close the MuJoCo window or press Esc to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = settled_ball_pose[:3]
        viewer.cam.distance = 0.35
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
