from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.envs.manipulation.allegro_inhand.grasp_gen import (
    AllegroRotationGrasp,
)
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    AllegroRotationPPO,
)
from unilab.envs.manipulation.leap_inhand.ball_grasp_allegro import (
    LeapAllegroGraspResetProvider,
    LeapInhandBallGraspAllegroCfg,
    LeapInhandBallGraspAllegroEnv,
)
from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import LeapInhandBallGraspEnv

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"
APPO_CONF_DIR = ROOT / "conf" / "appo"
SCENE = ROOT / "src" / "unilab" / "assets" / "robots" / "leap_hand" / "scene_ball.xml"
STRICT_CACHE = "robots/leap_hand/caches/ball_grasp_official_50k.npy"
NEW_CACHE = "robots/leap_hand/caches/ball_grasp_hora_sharpa_style_50k.npy"
SEED = np.asarray(
    [
        1.5152045040427635,
        0.11430147259750476,
        0.2876406730815961,
        0.19280835997306603,
        1.4188457206477074,
        0.025681830807677088,
        -0.26717932336688344,
        0.5369823550831088,
        1.5294890485315962,
        -0.01798386011739139,
        0.27558019211759954,
        0.19821762108233876,
        1.9245445859343515,
        0.04788276935232176,
        -0.021885380331691334,
        0.19524630120127295,
        -0.032440416893199604,
        0.041151239943936,
        0.664301098275159,
        0.9300906819767993,
        0.07052047191574277,
        -0.04548098804911446,
        0.3576166467976395,
    ],
    dtype=np.float64,
)


def _compose_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_grasp_allegro/mujoco"],
        )


def _compose_appo_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(APPO_CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_grasp_allegro/mujoco"],
        )


def _provider_env(*, noise: float = 0.15, object_scale: float = 1.0):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            grasp_seed_qpos=SEED.tolist(),
            object_scale=object_scale,
            domain_rand=SimpleNamespace(joint_noise=noise),
        ),
        _NUM_HAND_DOF=16,
        _ctrl_lower=np.full(16, -10.0),
        _ctrl_upper=np.full(16, 10.0),
        nv=22,
    )


def _cache_env():
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._cfg = SimpleNamespace()
    env._saved_grasping_states = []
    env._state = None
    return env


def _row() -> np.ndarray:
    row = np.zeros(23, dtype=np.float32)
    row[19] = 1.0
    return row


def _collector_env(tmp_path: Path, rows: np.ndarray):
    env = _cache_env()
    env._cfg.grasp_quality_check = False
    env._cfg.grasp_collection_target = 50_000
    env._cfg.grasp_cache_path = str(tmp_path / "unused.npy")
    env._cfg.grasp_auto_save = False
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(len(rows)),
        terminated=np.zeros(len(rows), dtype=bool),
        truncated=np.ones(len(rows), dtype=bool),
        info={
            "curr_dof_pos": rows[:, :16].copy(),
            "curr_ball_pos": rows[:, 16:19].copy(),
            "curr_ball_quat": rows[:, 19:23].copy(),
            "log": {},
        },
    )
    env._save_grasp_cache = lambda *args, **kwargs: None
    env._stop_collection = lambda: None
    env.get_hand_dof_pos = lambda: rows[:, :16]
    env.get_ball_pos = lambda: rows[:, 16:19]
    env.get_ball_quat = lambda: rows[:, 19:23]
    return env


def _load_inspector_module():
    path = ROOT / "scripts" / "inspect_leap_allegro_grasp_cache.py"
    spec = importlib.util.spec_from_file_location("inspect_leap_allegro_cache", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_registration_and_hydra_contract() -> None:
    registered = registry.list_registered_envs()
    assert "LeapInhandBallGraspAllegro" in registered
    assert registered["LeapInhandBallGraspAllegro"]["available_backends"] == ["mujoco"]

    cfg = _compose_cfg()
    seed = np.asarray(cfg.env.grasp_seed_qpos, dtype=np.float64)
    assert cfg.training.task_name == "LeapInhandBallGraspAllegro"
    assert seed.shape == (23,)
    assert np.isfinite(seed).all()
    np.testing.assert_array_equal(seed, SEED)
    assert np.linalg.norm(seed[19:23]) == pytest.approx(1.0)
    assert cfg.env.grasp_max_fingertip_distance == pytest.approx(0.1)
    assert cfg.env.grasp_max_height_delta == pytest.approx(0.005)
    assert cfg.env.grasp_max_orientation_error == pytest.approx(np.deg2rad(30.0))
    assert cfg.reward.reset_z_threshold == pytest.approx(0.0)
    assert cfg.env.max_episode_seconds == pytest.approx(3.0)
    assert int(round(cfg.env.max_episode_seconds / cfg.env.ctrl_dt)) == 60
    assert cfg.env.grasp_min_contacts == 2
    assert cfg.env.domain_rand.joint_noise == pytest.approx(0.15)
    assert cfg.env.grasp_cache_path == NEW_CACHE
    assert cfg.env.grasp_cache_path != STRICT_CACHE


def test_appo_owner_preserves_cache_generation_contract() -> None:
    ppo_cfg = _compose_cfg()
    appo_cfg = _compose_appo_cfg()

    assert appo_cfg.algo.algo == "appo"
    assert appo_cfg.training.task_name == "LeapInhandBallGraspAllegro"
    assert appo_cfg.training.sim_backend == "mujoco"
    assert appo_cfg.training.no_play is True
    assert appo_cfg.algo.num_envs == ppo_cfg.algo.num_envs == 1024
    assert appo_cfg.algo.steps_per_env == ppo_cfg.algo.num_steps_per_env == 8
    assert appo_cfg.algo.max_iterations == ppo_cfg.algo.max_iterations == 1_000_000
    assert appo_cfg.env == ppo_cfg.env
    assert appo_cfg.reward == ppo_cfg.reward


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"grasp_seed_qpos": [0.0] * 22}, "shape"),
        ({"grasp_seed_qpos": [0.0] * 23}, "non-zero"),
        ({"grasp_max_fingertip_distance": 0.0}, "positive"),
        ({"grasp_max_height_delta": 0.0}, "positive"),
        ({"grasp_max_orientation_error": 0.0}, "positive"),
        ({"grasp_collection_target": 0}, "positive"),
        ({"grasp_min_contacts": 5}, "within"),
    ],
)
def test_config_validation_rejects_invalid_values(overrides, match) -> None:
    cfg = LeapInhandBallGraspAllegroCfg(grasp_seed_qpos=SEED.tolist())
    for name, value in overrides.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_proposal_sampling_shapes_ranges_and_fixed_ball() -> None:
    provider = LeapAllegroGraspResetProvider()
    env = _provider_env()
    np.random.seed(123)

    hand_qpos, ball_pos, ball_quat, qvel = provider._sample_reset_state(env, 128)

    assert hand_qpos.shape == (128, 16)
    assert ball_pos.shape == (128, 3)
    assert ball_quat.shape == (128, 4)
    assert qvel.shape == (128, 22)
    offsets = hand_qpos - SEED[None, :16]
    assert np.all(offsets >= -0.15)
    assert np.all(offsets <= 0.15)
    np.testing.assert_array_equal(ball_pos, np.broadcast_to(SEED[16:19], (128, 3)))
    np.testing.assert_array_equal(ball_quat, np.broadcast_to(SEED[19:23], (128, 4)))
    assert not np.allclose(ball_quat, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(qvel, 0.0)


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2])
def test_all_scales_share_original_1_0_hand_and_object_pose(scale: float) -> None:
    provider = LeapAllegroGraspResetProvider()
    env = _provider_env(noise=0.0, object_scale=scale)

    hand_qpos, ball_pos, ball_quat, _ = provider._sample_reset_state(env, 3)

    np.testing.assert_array_equal(hand_qpos, np.broadcast_to(SEED[:16], (3, 16)))
    np.testing.assert_array_equal(ball_pos, np.broadcast_to(SEED[16:19], (3, 3)))
    np.testing.assert_array_equal(ball_quat, np.broadcast_to(SEED[19:23], (3, 4)))


def test_proposal_sampling_clips_joint_limits(monkeypatch) -> None:
    provider = LeapAllegroGraspResetProvider()
    env = _provider_env()
    env._ctrl_lower = SEED[:16] - 0.1
    env._ctrl_upper = SEED[:16] + 0.1
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda *args, **kwargs: np.full(kwargs["size"], 0.25),
    )

    hand_qpos, _, _, _ = provider._sample_reset_state(env, 3)

    np.testing.assert_allclose(
        hand_qpos,
        np.broadcast_to(env._ctrl_upper, hand_qpos.shape),
    )


def test_scale_collection_builds_one_physical_ball_variant() -> None:
    provider = LeapAllegroGraspResetProvider()
    env = SimpleNamespace(
        _num_envs=3,
        cfg=SimpleNamespace(object_scale=1.2),
        _backend=SimpleNamespace(get_geom_size=lambda name: np.asarray([0.0335, 0.0, 0.0])),
    )

    plan = provider.build_init_randomization_plan(env)

    np.testing.assert_array_equal(plan.model_assignments, [0, 0, 0])
    assert len(plan.model_variants) == 1
    override = plan.model_variants[0].geom_size_overrides[0]
    assert override.geom_name == "leap_object_col"
    np.testing.assert_allclose(override.size, [0.0402, 0.0, 0.0])


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_scale_collection_rejects_invalid_object_scale(value: float) -> None:
    cfg = LeapInhandBallGraspAllegroCfg(grasp_seed_qpos=SEED.tolist(), object_scale=value)

    with pytest.raises(ValueError, match="object_scale"):
        cfg.validate()


def test_inherited_info_updates_set_prev_ctrl_from_sampled_qpos() -> None:
    provider = LeapAllegroGraspResetProvider()
    hand = np.broadcast_to(SEED[:16], (2, 16)).copy()
    hand[1, 0] += 0.1
    ball_pos = np.broadcast_to(SEED[16:19], (2, 3)).copy()
    ball_quat = np.broadcast_to(SEED[19:23], (2, 4)).copy()
    env = SimpleNamespace(
        _dof_mid=np.zeros(16),
        _dof_range=np.ones(16),
        _NUM_LAG_STEPS=3,
        _NUM_OBS_PER_STEP=35,
        _num_action=16,
    )

    updates = provider._build_info_updates(env, hand, ball_pos, ball_quat)

    np.testing.assert_allclose(updates["prev_ctrl"], hand, rtol=1e-6)
    np.testing.assert_allclose(updates["init_pose"], hand, rtol=1e-6)
    np.testing.assert_allclose(updates["prev_dof_pos"], hand, rtol=1e-6)
    np.testing.assert_allclose(updates["prev_ball_pos"], ball_pos, rtol=1e-6)
    np.testing.assert_allclose(updates["prev_ball_quat"], ball_quat, rtol=1e-6)
    np.testing.assert_allclose(updates["initial_ball_z"], ball_pos[:, 2], rtol=1e-6)
    assert not np.shares_memory(updates["initial_ball_z"], ball_pos)
    np.testing.assert_allclose(updates["initial_ball_quat"], ball_quat, rtol=1e-6)
    assert not np.shares_memory(updates["initial_ball_quat"], ball_quat)
    np.testing.assert_array_equal(updates["current_actions"], 0.0)
    np.testing.assert_array_equal(updates["last_actions"], 0.0)


def test_external_actions_are_ignored_and_sampled_target_is_held() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._num_envs = 2
    env._num_action = 16
    env._np_dtype = np.float32
    env.default_angles = np.zeros(16, dtype=np.float32)
    env._ctrl_lower = np.full(16, -10.0)
    env._ctrl_upper = np.full(16, 10.0)
    env._cfg = SimpleNamespace(control_config=SimpleNamespace(action_scale=1.0 / 24.0))
    sampled = np.stack([SEED[:16], SEED[:16] + 0.01]).astype(np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        info={"prev_ctrl": sampled.copy()},
    )

    ctrl = env.apply_action(np.ones((2, 16), dtype=np.float32), state)

    np.testing.assert_array_equal(ctrl, sampled)
    np.testing.assert_array_equal(state.info["current_actions"], 0.0)


def test_four_sharpa_conditions_use_strict_boundaries() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._num_envs = 1
    env._cfg = SimpleNamespace(
        grasp_max_fingertip_distance=0.1,
        grasp_min_contacts=2,
        grasp_max_height_delta=0.005,
        grasp_max_orientation_error=np.deg2rad(30.0),
    )
    initial_ball_z = np.float32(SEED[18])
    initial_ball_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(1),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={
            "initial_ball_z": np.asarray([initial_ball_z]),
            "initial_ball_quat": initial_ball_quat,
        },
    )
    lower_threshold = np.float32(initial_ball_z - np.float32(0.005))
    ball = np.asarray([[0.0, 0.0, lower_threshold]], dtype=np.float32)
    tips = np.asarray([[[0.1, 0.0, ball[0, 2]]] * 4])
    ball_quat = np.asarray(
        [[np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0, 0.0]],
        dtype=np.float32,
    )
    env.get_ball_pos = lambda: ball
    env.get_ball_quat = lambda: ball_quat
    env.get_fingertip_pos = lambda: tips
    env._contact_count = lambda: np.asarray([1], dtype=np.int32)

    cond1, cond2, cond3, cond4 = env._compute_sharpa_grasp_conditions()

    assert not cond1[0]
    assert not cond2[0]
    assert not cond3[0]
    assert not cond4[0]

    tips[:, :, 0] = 0.09999
    ball[:, 2] = np.nextafter(ball[:, 2], np.float32(np.inf))
    ball_quat[:] = [
        np.cos(np.deg2rad(14.999)),
        np.sin(np.deg2rad(14.999)),
        0.0,
        0.0,
    ]
    env._contact_count = lambda: np.asarray([2], dtype=np.int32)
    conditions = env._compute_sharpa_grasp_conditions()
    assert all(condition[0] for condition in conditions)

    ball[:, 2] = np.float32(initial_ball_z + np.float32(0.005))
    assert not env._compute_sharpa_grasp_conditions()[2][0]


def test_quality_gate_requires_all_four_sharpa_conditions() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    conditions = (
        np.asarray([True, True]),
        np.asarray([True, False]),
        np.asarray([True, True]),
        np.asarray([True, True]),
    )
    env._compute_sharpa_grasp_conditions = lambda: conditions

    valid = env._check_grasp_quality(np.asarray([0, 1], dtype=np.int32))

    np.testing.assert_array_equal(valid, [True, False])


def test_contact_count_allows_index_middle_without_thumb_and_ignores_palm() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    sensor_values = {
        "leap_index_contact": np.asarray([[1.0]]),
        "leap_middle_contact": np.asarray([[1.0]]),
        "leap_ring_contact": np.asarray([[0.0]]),
        "leap_thumb_contact": np.asarray([[0.0]]),
        "leap_palm_contact": np.asarray([[1.0]]),
    }
    env.get_sensor_data = lambda name: sensor_values[name]

    count = env._contact_count()

    np.testing.assert_array_equal(count, [2])
    assert "leap_palm_contact" not in env._CONTACT_SENSORS


def test_first_update_has_no_warmup_and_terminates(monkeypatch) -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._num_envs = 1
    env._np_dtype = np.float32
    env._enable_reward_log = False
    env._cfg = SimpleNamespace(grasp_quality_check=True)
    env._compute_grasp_conditions = lambda: (
        np.asarray([False]),
        np.asarray([True]),
        np.asarray([True]),
    )
    state = NpEnvState(
        obs={},
        reward=np.ones(1),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={"steps": np.zeros(1, dtype=np.uint32)},
    )
    monkeypatch.setattr(AllegroRotationPPO, "update_state", lambda self, value: value)

    result = AllegroRotationGrasp.update_state(env, state)

    assert result.terminated[0]
    np.testing.assert_array_equal(result.reward, 0.0)


def test_timeout_success_collector_uses_final_settled_rows(tmp_path) -> None:
    rows = np.stack([_row(), _row(), _row()])
    rows[0, 0] = 0.123
    rows[1, 0] = 0.456
    rows[2, 0] = 0.789
    env = _collector_env(tmp_path, rows)
    env.state.terminated[1] = True
    env.state.truncated[2] = False

    env._collect_successful_grasps(np.asarray([0, 1, 2], dtype=np.int32))

    assert env._total_saved_grasps() == 1
    assert env._saved_grasping_states[0].dtype == np.float32
    assert env._saved_grasping_states[0].shape == (1, 23)
    assert env._saved_grasping_states[0][0, 0] == pytest.approx(0.123)


def test_leap_sharpa_filter_is_identity_and_keeps_duplicates() -> None:
    env = _cache_env()
    rows = np.stack([_row(), _row()])

    kept = env._filter_grasp_rows(rows)

    assert kept is rows


def test_allegro_default_filter_is_identity_and_collector_keeps_duplicates(tmp_path) -> None:
    rows = np.stack([_row(), _row()])
    env = object.__new__(AllegroRotationGrasp)
    assert env._filter_grasp_rows(rows) is rows
    env._cfg = SimpleNamespace(
        grasp_quality_check=False,
        grasp_collection_target=50_000,
        grasp_cache_path=str(tmp_path / "unused.npy"),
        grasp_auto_save=False,
    )
    env._saved_grasping_states = []
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.ones(2, dtype=bool),
        info={
            "curr_dof_pos": rows[:, :16],
            "curr_ball_pos": rows[:, 16:19],
            "curr_ball_quat": rows[:, 19:23],
            "log": {},
        },
    )
    env._save_grasp_cache = lambda *args, **kwargs: None
    env._stop_collection = lambda: None
    env.get_hand_dof_pos = lambda: rows[:, :16]
    env.get_ball_pos = lambda: rows[:, 16:19]
    env.get_ball_quat = lambda: rows[:, 19:23]

    env._collect_successful_grasps(np.asarray([0, 1], dtype=np.int32))

    assert env._total_saved_grasps() == 2


def test_new_environment_does_not_define_strict_leap_paths() -> None:
    forbidden_attributes = {
        "grasp_require_thumb_contact",
        "grasp_warmup_seconds",
        "grasp_max_ball_drift",
        "grasp_max_ball_linear_speed",
        "grasp_max_ball_angular_speed",
        "grasp_max_joint_speed",
        "grasp_max_abs_work",
        "grasp_max_self_penetration",
        "grasp_max_object_penetration",
        "grasp_frontier_fraction",
        "grasp_max_fingertip_surface_gap",
        "grasp_dedup_enabled",
        "grasp_dedup_joint_resolution",
        "grasp_dedup_ball_position_resolution",
    }
    cfg_fields = LeapInhandBallGraspAllegroCfg.__dataclass_fields__
    assert forbidden_attributes.isdisjoint(cfg_fields)
    assert "_strict_quality_mask" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "_penetration_quality" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "replay_validate_grasp_cache_rows" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "update_state" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert not issubclass(LeapInhandBallGraspAllegroEnv, LeapInhandBallGraspEnv)
    source = inspect.getsource(LeapAllegroGraspResetProvider._sample_reset_state)
    assert "frontier" not in source
    assert "ball_position_noise" not in source
    environment_source = inspect.getsource(LeapInhandBallGraspAllegroEnv)
    for forbidden_path in (
        "_backend.set_state",
        "_penetration_quality",
        "replay_validate_grasp_cache_rows",
        "_strict_quality_mask",
        "grasp_require_thumb_contact",
    ):
        assert forbidden_path not in environment_source


def test_direct_save_uses_float32_rows_and_temporary_path_without_dedup(tmp_path) -> None:
    env = _cache_env()
    rows = np.stack([_row(), _row(), _row()])
    rows[2, 0] = 0.02
    env._saved_grasping_states.append(env._filter_grasp_rows(rows))
    output = tmp_path / "cache.npy"
    env._cfg.grasp_cache_path = str(output)
    env._cfg.grasp_collection_target = 2
    env._grasp_cache_saved = False

    AllegroRotationGrasp._save_grasp_cache(env, force=True)

    saved = np.load(output)
    assert saved.shape == (2, 23)
    assert saved.dtype == np.float32
    assert not output.with_suffix(".npy.tmp").exists()
    assert env._last_grasp_auto_save_total == 2
    assert output.name != Path(STRICT_CACHE).name
    np.testing.assert_array_equal(saved[0], saved[1])


def test_partial_scale_cache_is_restored_for_resume(tmp_path) -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    rows = np.stack([_row(), _row()]).astype(np.float32)
    rows[1, 0] = 0.1
    cache = tmp_path / "partial.npy"
    np.save(cache, rows)
    env._cfg = SimpleNamespace(
        grasp_auto_save=True,
        grasp_cache_path=str(cache),
        grasp_collection_target=50_000,
    )
    env._saved_grasping_states = []
    env._grasp_cache_saved = False
    env._last_grasp_auto_save_total = 0

    env._restore_partial_grasp_cache()

    assert env._total_saved_grasps() == 2
    assert env._last_grasp_auto_save_total == 2
    assert env._grasp_cache_saved is True


def test_cache_inspector_reports_quantized_and_quaternion_only_duplicates(tmp_path) -> None:
    inspector = _load_inspector_module()
    rows = np.stack([_row(), _row(), _row()])
    rows[1, 19:23] = [0.0, 1.0, 0.0, 0.0]
    rows[2, 0] = 0.02
    path = tmp_path / "inspect.npy"
    np.save(path, rows)

    report = inspector.inspect_cache(
        path,
        expected_rows=3,
        joint_resolution=0.001,
        ball_position_resolution=0.0005,
        nominal_ball_z=SEED[18],
        max_drop_distance=0.005,
    )

    assert report["file_exists"]
    assert report["shape"] == [3, 23]
    assert report["dtype"] == "float32"
    assert report["dtype_valid"]
    assert report["finite"]
    assert report["exact_duplicate_rows"] == 0
    assert report["quantized_unique_key_count"] == 2
    assert report["quantized_duplicate_count"] == 1
    assert report["quaternion_only_duplicate_group_count"] == 1
    assert report["expected_row_count_pass"]


def test_cache_inspector_rejects_rows_dropped_5_mm_from_nominal(tmp_path) -> None:
    inspector = _load_inspector_module()
    rows = np.stack([_row(), _row()])
    threshold = float(np.float32(SEED[18] - 0.005))
    rows[0, 18] = threshold + 1e-6
    rows[1, 0] = 0.01
    rows[1, 18] = threshold
    path = tmp_path / "height_gate.npy"
    np.save(path, rows)

    report = inspector.inspect_cache(
        path,
        expected_rows=2,
        joint_resolution=0.001,
        ball_position_resolution=0.0005,
        nominal_ball_z=SEED[18],
        max_drop_distance=0.005,
    )

    assert report["height_threshold"] == pytest.approx(threshold)
    assert report["height_rejected_row_count"] == 1
    assert not report["height_valid"]


def test_leap_ball_collision_radius_is_33_5_mm() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leap_object_col")
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert model.geom_size[geom_id, 0] == pytest.approx(0.0335)


def test_reset_provider_extends_allegro_info_with_initial_ball_height() -> None:
    assert issubclass(
        LeapAllegroGraspResetProvider,
        AllegroRotationDomainRandomizationProvider,
    )
    assert "_build_info_updates" in LeapAllegroGraspResetProvider.__dict__
