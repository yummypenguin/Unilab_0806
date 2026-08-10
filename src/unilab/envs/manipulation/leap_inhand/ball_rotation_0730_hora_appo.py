"""Sharpa-parity HORA APPO teacher task for the LEAP Hand ball rotation task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_grasp_cache_files
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.manipulation.sharpa_inhand.base import (
    SharpaControlConfig,
    SharpaDomainRandConfig,
    SharpaObservationConfig,
    SharpaPrivilegedInfoConfig,
    SharpaSensorConfig,
    resolve_grasp_cache_file,
)
from unilab.envs.manipulation.sharpa_inhand.rotation import (
    RewardConfig,
    SharpaInhandRotationCfg,
    SharpaInhandRotationDRProvider,
    SharpaInhandRotationEnv,
)

from .ball_rotation_0730 import _CACHE_GENERATION_NOMINAL_HAND_QPOS
from .base import MENAGERIE_SIM_JOINT_NAMES
from .hora_appo_deploy_contract import build_deploy_contract_manifest

LEAP_TACTILE_FORCE_SENSOR_NAMES: tuple[str, ...] = (
    "leap_index_tactile_force",
    "leap_middle_tactile_force",
    "leap_ring_tactile_force",
    "leap_thumb_tactile_force",
)

# Authoritative 23D seed used by LeapInhandBallGraspAllegro to generate
# ball_grasp_hora_sharpa_style_50k.npy. Keep the reward pose and object
# anchor tied to the same cache-generation nominal setup.
_CACHE_GENERATION_NOMINAL_BALL_POSE: tuple[float, ...] = (
    -0.032440416893199604,
    0.041151239943936,
    0.664301098275159,
    0.9300906819767993,
    0.07052047191574277,
    -0.04548098804911446,
    0.3576166467976395,
)


@registry.envcfg("LeapInhandBall0730HoraAppoRotation")
@dataclass
class LeapInhandBall0730HoraAppoRotationCfg(SharpaInhandRotationCfg):
    """LEAP embodiment of the Sharpa HORA APPO rotation contract."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml")
        )
    )
    sim_dt: float = 0.005
    ctrl_dt: float = 0.05
    max_episode_seconds: float = 20.0

    action_space: int = 16
    observation_space: int = 96
    num_hand_dofs: int = 16
    frame_obs_dim: int = 32
    obs_lag_steps: int = 3
    obs_history_len: int = 80
    prop_hist_len: int = 30
    critic_info_dim: int = 9

    base_name: str = "palm_lower"
    object_body_name: str = "leap_object"
    object_geom_name: str = "leap_object_col"
    object_visual_mesh_name: str | None = "leap_ball_visual_mesh"
    actuated_joint_names: list[str] = field(default_factory=lambda: list(MENAGERIE_SIM_JOINT_NAMES))
    fingertip_body_names: list[str] = field(
        default_factory=lambda: [
            "fingertip",
            "fingertip_2",
            "fingertip_3",
            "thumb_fingertip",
        ]
    )
    default_hand_joint_pos: list[float] = field(
        default_factory=lambda: list(_CACHE_GENERATION_NOMINAL_HAND_QPOS)
    )
    default_object_pose: list[float] = field(
        default_factory=lambda: list(_CACHE_GENERATION_NOMINAL_BALL_POSE)
    )
    tactile_diagnostic_names: list[str] = field(
        default_factory=lambda: ["index", "middle", "ring", "thumb"]
    )

    sensor: SharpaSensorConfig = field(  # type: ignore[assignment]
        default_factory=lambda: SharpaSensorConfig(
            tactile_force_sensor_names=list(LEAP_TACTILE_FORCE_SENSOR_NAMES)
        )
    )
    obs: SharpaObservationConfig = field(
        default_factory=lambda: SharpaObservationConfig(
            observation_mode="separated",
            # Tactile force sensors are simulation-only diagnostics for this
            # LEAP embodiment. They are intentionally excluded from the
            # deployable HORA actor/proprio observation contract.
            enable_tactile=False,
            binary_contact=False,
            enable_contact_pos=False,
            contact_smooth=0.5,
            contact_threshold=0.05,
            tactile_force_clip_max=4.0,
        )
    )
    priv_info: SharpaPrivilegedInfoConfig = field(
        default_factory=lambda: SharpaPrivilegedInfoConfig(
            include_friction_scale=True,
            include_gravity_direction=False,
        )
    )
    control_config: SharpaControlConfig = field(
        default_factory=lambda: SharpaControlConfig(
            action_scale=1.0 / 24.0,
            p_gain=3.0,
            d_gain=0.01,
            torque_control=False,
            dof_limits_scale=1.0,
        )
    )
    domain_rand: SharpaDomainRandConfig = field(
        default_factory=lambda: SharpaDomainRandConfig(
            scale_list=[0.8, 1.0, 1.2],
            randomize_gravity_direction=True,
            gravity_direction_magnitude=9.81,
            gravity_direction_tilt_max_deg=3.0,
            randomize_pd_gains=True,
            randomize_p_gain_scale_lower=2.9 / 3.0,
            randomize_p_gain_scale_upper=3.1 / 3.0,
            randomize_d_gain_scale_lower=0.9,
            randomize_d_gain_scale_upper=1.1,
            randomize_friction=True,
            randomize_friction_scale_lower=0.3,
            randomize_friction_scale_upper=3.0,
            scale_xml_friction_per_geom=True,
            randomize_com=True,
            randomize_com_lower=-0.01,
            randomize_com_upper=0.01,
            randomize_mass=True,
            randomize_mass_lower=0.01,
            randomize_mass_upper=0.25,
            force_scale=2.0,
            random_force_prob_scalar=0.25,
            force_decay=0.9,
            force_decay_interval=0.08,
            joint_noise_scale=0.0,
            contact_latency=0.005,
            contact_sensor_noise=0.01,
        )
    )

    reward_config: RewardConfig | None = None
    rot_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # The shared Sharpa reset provider uses this range's width and centers it
    # on each sampled cache row. A 0.06 m width therefore means +/-30 mm.
    reset_height_lower: float = 0.58906
    reset_height_upper: float = 0.64906
    grasp_cache_path: str = "robots/leap_hand/caches/ball_grasp_hora_sharpa_style_50k"
    use_default_object_pose_for_object_pos_anchor: bool = True


class LeapInhandBall0730HoraAppoDRProvider(SharpaInhandRotationDRProvider):
    """LEAP cache owner for the shared Sharpa-parity reset/DR contract."""

    def _load_grasp_cache(self, env: Any) -> tuple[np.ndarray, ...]:
        cached = getattr(env, "_grasp_cache", None)
        if cached is not None:
            return cast(tuple[np.ndarray, ...], cached)

        grasp_caches: list[np.ndarray] = []
        missing_files: list[str] = []
        for scale_value in np.asarray(env.scale_values, dtype=np.float64):
            cache_file = resolve_grasp_cache_file(env.cfg.grasp_cache_path, float(scale_value))
            resolved = Path(cast(str, resolve_grasp_cache_files(str(cache_file))))
            if not resolved.exists():
                missing_files.append(str(resolved))
                continue
            cache = np.load(resolved).astype(np.float64)
            expected_width = int(env._num_action) + 7
            if cache.ndim != 2 or cache.shape[1] != expected_width:
                raise ValueError(
                    f"Expected LEAP grasp cache shape (?, {expected_width}), got {cache.shape}"
                )
            grasp_caches.append(cache)

        if missing_files:
            missing = "\n  ".join(missing_files)
            raise RuntimeError(
                "Missing LEAP scale-specific grasp cache file(s):\n  "
                f"{missing}\n"
                "Generate every configured scale cache with:\n"
                "  bash scripts/leap_collect_hora_grasps.sh "
                "0.8 1.0 1.2\n"
                "scale=1.0 cache fallback is intentionally disabled."
            )

        env._grasp_cache = tuple(grasp_caches)
        return cast(tuple[np.ndarray, ...], env._grasp_cache)


@registry.env("LeapInhandBall0730HoraAppoRotation", sim_backend="mujoco")
class LeapInhandBall0730HoraAppoRotationEnv(SharpaInhandRotationEnv):
    """LEAP-adapted environment with Sharpa HORA APPO task semantics."""

    _cfg: LeapInhandBall0730HoraAppoRotationCfg

    def __init__(
        self,
        cfg: LeapInhandBall0730HoraAppoRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        dr_provider = LeapInhandBall0730HoraAppoDRProvider()
        super().__init__(
            cfg,
            num_envs=num_envs,
            backend_type=backend_type,
            dr_provider=dr_provider,
        )
        # Asset/cache resolution belongs to construction, never reset/step.
        # This also makes an incomplete scale set fail before training starts.
        self._grasp_cache = dr_provider._load_grasp_cache(self)

    def build_deploy_contract_manifest(self) -> dict[str, Any]:
        """Export no-tactile policy metadata from the materialized model contract."""

        return build_deploy_contract_manifest(
            joint_names=self._cfg.actuated_joint_names,
            joint_lower=self._ctrl_lower,
            joint_upper=self._ctrl_upper,
        )

    def _read_tactile_force(self) -> np.ndarray:
        """Read simulation-only tactile diagnostics, never policy inputs."""
        sensor_names = tuple(self._cfg.sensor.tactile_force_sensor_names)
        if len(sensor_names) != self._num_tactile:
            raise ValueError(
                "LEAP HORA APPO requires exactly one force sensor per fingertip; "
                f"got {len(sensor_names)} sensors for {self._num_tactile} fingertips"
            )
        tactile_force = np.zeros((self._num_envs, self._num_tactile), dtype=self._np_dtype)
        for sensor_id, sensor_name in enumerate(sensor_names):
            # Fail closed: a missing or malformed force sensor is a model-contract
            # error and must never be replaced with a silent zero channel.
            tactile_force[:, sensor_id] = self._extract_sensor_scalar(sensor_name)
        return tactile_force

    def _resolve_friction_geom_ids(self) -> dict[str, np.ndarray]:
        object_geom_id = self._backend.get_geom_id(self._cfg.object_geom_name)
        base_body_id = self._backend.get_body_id(self._cfg.base_name)
        hand_body_ids = set(
            int(body_id) for body_id in self._backend.get_body_subtree_ids(base_body_id)
        )
        fingertip_body_ids: set[int] = set()
        for body_id in np.asarray(self._fingertip_body_ids, dtype=np.int32):
            fingertip_body_ids.update(
                int(value) for value in self._backend.get_body_subtree_ids(int(body_id))
            )

        geom_body_ids = self._backend.get_geom_body_ids()
        geom_contype, geom_conaffinity = self._backend.get_geom_contact_masks()
        fingertip_ids: list[int] = []
        hand_ids: list[int] = []
        for geom_id, body_id in enumerate(geom_body_ids):
            body_id_int = int(body_id)
            if body_id_int not in hand_body_ids:
                continue
            if int(geom_contype[geom_id]) == 0 and int(geom_conaffinity[geom_id]) == 0:
                continue
            if body_id_int in fingertip_body_ids:
                fingertip_ids.append(geom_id)
            else:
                hand_ids.append(geom_id)

        if not fingertip_ids:
            raise ValueError("No LEAP fingertip collision geoms found for friction randomization")
        if not hand_ids:
            raise ValueError(
                "No LEAP non-fingertip hand collision geoms found for friction randomization"
            )
        return {
            "object": np.asarray([object_geom_id], dtype=np.int32),
            "elastomer": np.asarray(fingertip_ids, dtype=np.int32),
            "metal": np.asarray(hand_ids, dtype=np.int32),
        }


__all__ = [
    "LEAP_TACTILE_FORCE_SENSOR_NAMES",
    "LeapInhandBall0730HoraAppoDRProvider",
    "LeapInhandBall0730HoraAppoRotationCfg",
    "LeapInhandBall0730HoraAppoRotationEnv",
]
