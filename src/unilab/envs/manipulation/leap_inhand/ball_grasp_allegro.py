"""Sharpa-style LEAP HORA ball-grasp cache generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.dr import (
    DomainRandomizationProvider,
    GeomSizeOverride,
    InitRandomizationPlan,
    ModelVariantSpec,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.grasp_gen import AllegroRotationGrasp
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
)
from unilab.utils.rotation import np_quat_error_magnitude

from .ball_rotation import LeapInhandBallRotationCfg
from .base import LeapHandBaseEnv


@registry.envcfg("LeapInhandBallGraspAllegro")
@dataclass
class LeapInhandBallGraspAllegroCfg(LeapInhandBallRotationCfg):
    """Configuration for Allegro-lifecycle LEAP grasp collection."""

    gen_grasp: bool = True
    grasp_cache_path: str = "robots/leap_hand/caches/ball_grasp_hora_sharpa_style_50k.npy"
    grasp_collection_target: int = 50_000
    grasp_auto_save: bool = True
    grasp_auto_save_interval: int = 1_000
    grasp_quality_check: bool = True
    grasp_min_contacts: int = 2
    grasp_seed_qpos: list[float] = field(default_factory=list)
    grasp_max_fingertip_distance: float = 0.1
    grasp_max_height_delta: float = 0.005
    grasp_max_orientation_error: float = np.deg2rad(30.0)
    # One collection run owns exactly one physical object scale. The output
    # cache path must be selected explicitly by the launch command so scale
    # buckets can never overwrite or silently share one cache.
    object_scale: float = 1.0

    def validate(self) -> None:
        super().validate()
        seed = np.asarray(self.grasp_seed_qpos, dtype=np.float64)
        if seed.shape != (23,):
            raise ValueError(f"grasp_seed_qpos must have shape (23,), got {seed.shape}")
        if not np.isfinite(seed).all():
            raise ValueError("grasp_seed_qpos must contain only finite values")
        if float(np.linalg.norm(seed[19:23])) <= 1e-8:
            raise ValueError("grasp_seed_qpos quaternion must have non-zero length")
        if (
            not np.isfinite(self.grasp_max_fingertip_distance)
            or self.grasp_max_fingertip_distance <= 0.0
        ):
            raise ValueError("grasp_max_fingertip_distance must be positive and finite")
        if self.grasp_collection_target <= 0:
            raise ValueError("grasp_collection_target must be positive")
        if not np.isfinite(self.grasp_max_height_delta) or self.grasp_max_height_delta <= 0.0:
            raise ValueError("grasp_max_height_delta must be positive and finite")
        if (
            not np.isfinite(self.grasp_max_orientation_error)
            or self.grasp_max_orientation_error <= 0.0
        ):
            raise ValueError("grasp_max_orientation_error must be positive and finite")
        if not 0 <= self.grasp_min_contacts <= 4:
            raise ValueError("grasp_min_contacts must be within [0, 4]")
        if not np.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("object_scale must be positive and finite")


class LeapAllegroGraspResetProvider(AllegroRotationDomainRandomizationProvider):
    """Sample LEAP hand proposals around the task-owned 23D nominal seed."""

    def build_init_randomization_plan(self, env: Any) -> InitRandomizationPlan:
        """Compile one ball-size model variant for this collection run."""

        scale = float(env.cfg.object_scale)
        base_size = np.asarray(env._backend.get_geom_size("leap_object_col"), dtype=np.float64)
        return InitRandomizationPlan(
            model_assignments=np.zeros(env._num_envs, dtype=np.int32),
            model_variants=(
                ModelVariantSpec(
                    geom_size_overrides=(
                        GeomSizeOverride(
                            geom_name="leap_object_col",
                            size=tuple(np.asarray(base_size * scale, dtype=np.float64)),
                        ),
                    )
                ),
            ),
        )

    def _sample_reset_state(
        self, env: Any, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = np.asarray(env.cfg.grasp_seed_qpos, dtype=np.float64)
        hand_qpos = np.broadcast_to(seed[:16], (num_reset, 16)).copy()
        joint_noise = float(env.cfg.domain_rand.joint_noise)
        hand_qpos += np.random.uniform(
            -joint_noise,
            joint_noise,
            size=(num_reset, env._NUM_HAND_DOF),
        )
        hand_qpos = np.clip(
            hand_qpos,
            np.asarray(env._ctrl_lower, dtype=np.float64),
            np.asarray(env._ctrl_upper, dtype=np.float64),
        )
        ball_pos = np.broadcast_to(seed[16:19], (num_reset, 3)).copy()
        ball_quat = np.broadcast_to(seed[19:23], (num_reset, 4)).copy()
        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)
        return hand_qpos, ball_pos, ball_quat, qvel

    def _build_info_updates(
        self,
        env: Any,
        hand_qpos: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        updates = super()._build_info_updates(env, hand_qpos, ball_pos, ball_quat)
        updates["initial_ball_z"] = np.asarray(
            ball_pos[:, 2],
            dtype=get_global_dtype(),
        ).copy()
        updates["initial_ball_quat"] = np.asarray(
            ball_quat,
            dtype=get_global_dtype(),
        ).copy()
        return updates


@registry.env("LeapInhandBallGraspAllegro", sim_backend="mujoco")
class LeapInhandBallGraspAllegroEnv(AllegroRotationGrasp, LeapHandBaseEnv):
    """Collect fixed-target grasps that survive the four Sharpa acceptance gates."""

    _cfg: LeapInhandBallGraspAllegroCfg
    _CONTACT_SENSORS = (
        "leap_index_contact",
        "leap_middle_contact",
        "leap_ring_contact",
        "leap_thumb_contact",
    )

    def __init__(
        self,
        cfg: LeapInhandBallGraspAllegroCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._restore_partial_grasp_cache()

    def _restore_partial_grasp_cache(self) -> None:
        """Resume an interrupted scale-cache collection from its atomic autosave."""

        if not bool(self._cfg.grasp_auto_save):
            return
        cache_file = Path(self._cfg.grasp_cache_path)
        if not cache_file.is_absolute():
            cache_file = ASSETS_ROOT_PATH / cache_file
        if not cache_file.exists():
            return

        rows = np.asarray(np.load(cache_file), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self._NUM_HAND_DOF + 7:
            raise ValueError(f"Cannot resume invalid LEAP grasp cache shape {rows.shape}")
        target = int(self._cfg.grasp_collection_target)
        if target > 0:
            rows = rows[:target]
        self._saved_grasping_states = [rows.copy()]
        self._last_grasp_auto_save_total = int(rows.shape[0])
        self._grasp_cache_saved = True
        print(f"[Leap grasp cache] Resumed {rows.shape[0]} rows from {cache_file}")

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapAllegroGraspResetProvider()

    def _compute_sharpa_grasp_conditions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ball_pos = self.get_ball_pos()
        fingertip_pos = self.get_fingertip_pos()

        if self.state is None:
            raise RuntimeError("environment state is unavailable during grasp validation")
        initial_ball_z = np.asarray(
            self.state.info.get("initial_ball_z"),
            dtype=get_global_dtype(),
        )
        if initial_ball_z.shape != (self._num_envs,):
            raise RuntimeError("initial_ball_z must be initialized for every environment at reset")

        cond1 = np.all(
            np.linalg.norm(fingertip_pos - ball_pos[:, None, :], axis=-1)
            < float(self._cfg.grasp_max_fingertip_distance),
            axis=1,
        )
        cond2 = self._contact_count() >= int(self._cfg.grasp_min_contacts)
        initial_ball_quat = np.asarray(
            self.state.info.get("initial_ball_quat"),
            dtype=get_global_dtype(),
        )
        if initial_ball_quat.shape != (self._num_envs, 4):
            raise RuntimeError(
                "initial_ball_quat must be initialized for every environment at reset"
            )
        height_delta = float(self._cfg.grasp_max_height_delta)
        cond3 = (ball_pos[:, 2] > initial_ball_z - height_delta) & (
            ball_pos[:, 2] < initial_ball_z + height_delta
        )
        quat_error = np_quat_error_magnitude(initial_ball_quat, self.get_ball_quat())
        cond4 = quat_error < float(self._cfg.grasp_max_orientation_error)
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/height_valid"] = float(np.mean(cond3.astype(np.float32)))
            log["grasp/orientation_valid"] = float(np.mean(cond4.astype(np.float32)))
            self.state.info["log"] = log
        return (
            np.asarray(cond1, dtype=bool),
            np.asarray(cond2, dtype=bool),
            np.asarray(cond3, dtype=bool),
            np.asarray(cond4, dtype=bool),
        )

    def _compute_grasp_conditions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cond1, cond2, cond3, cond4 = self._compute_sharpa_grasp_conditions()
        return cond1, cond2, np.asarray(cond3 & cond4, dtype=bool)

    def _check_grasp_quality(self, env_ids: np.ndarray) -> np.ndarray:
        conditions = self._compute_sharpa_grasp_conditions()
        return np.asarray(
            np.logical_and.reduce([condition[env_ids] for condition in conditions]),
            dtype=bool,
        )


LeapInhandBallGraspAllegro = LeapInhandBallGraspAllegroEnv
