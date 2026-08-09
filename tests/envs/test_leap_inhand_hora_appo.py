from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora_appo import (
    LEAP_TACTILE_FORCE_SENSOR_NAMES,
    LeapInhandBall0730HoraAppoDRProvider,
    LeapInhandBall0730HoraAppoRotationCfg,
    LeapInhandBall0730HoraAppoRotationEnv,
)
from unilab.envs.manipulation.sharpa_inhand.base import (
    SharpaDomainRandConfig,
    SharpaInhandBaseCfg,
    SharpaInhandBaseEnv,
    resolve_grasp_cache_file,
)
from unilab.envs.manipulation.sharpa_inhand.rotation import (
    RewardConfig,
    SharpaInhandRotationEnv,
)

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf" / "appo"
SCENE = ROOT / "src" / "unilab" / "assets" / "robots" / "leap_hand" / "scene_ball.xml"


def _compose_hora_appo():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_0730/mujoco_hora"],
        )


def test_hora_appo_owner_composes_sharpa_parity_contract() -> None:
    cfg = _compose_hora_appo()

    assert cfg.training.task_name == "LeapInhandBall0730HoraAppoRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.runtime_impl == "hora_appo"
    assert cfg.algo.runtime_resolver == "unilab.algos.torch.hora.appo:resolve_hora_appo_runtime"
    assert cfg.algo.actor.class_name == "unilab.algos.torch.hora:HoraActorModel"
    assert cfg.algo.critic.class_name == "unilab.algos.torch.hora:HoraCriticModel"
    assert cfg.algo.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert cfg.algo.algorithm.desired_kl == pytest.approx(0.04)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.01)
    assert dict(cfg.reward.scales) == {
        "rotate": 2.5,
        "obj_linvel": -0.3,
        "pose_diff": -0.4,
        "torque": -0.1,
        "work": -0.5,
        "object_pos": 0.003,
    }
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.ctrl_dt == pytest.approx(0.05)
    assert cfg.env.ctrl_dt / cfg.env.sim_dt == pytest.approx(10.0)
    assert cfg.env.control_config.action_scale == pytest.approx(1.0 / 24.0)
    assert cfg.env.control_config.p_gain == pytest.approx(3.0)
    assert cfg.env.control_config.d_gain == pytest.approx(0.01)
    assert cfg.env.control_config.dof_limits_scale == pytest.approx(1.0)
    assert cfg.env.reset_height_upper - cfg.env.reset_height_lower == pytest.approx(0.06)
    assert cfg.env.obs.observation_mode == "separated"
    assert cfg.env.sensor.tactile_force_sensor_names == list(LEAP_TACTILE_FORCE_SENSOR_NAMES)
    assert cfg.env.object_visual_mesh_name == "leap_ball_visual_mesh"
    assert cfg.env.domain_rand.scale_list == [0.8, 1.0, 1.2]
    assert cfg.env.domain_rand.randomize_gravity_direction is True
    assert cfg.env.domain_rand.gravity_direction_magnitude == pytest.approx(9.81)
    assert cfg.env.domain_rand.gravity_direction_tilt_max_deg == pytest.approx(3.0)
    assert cfg.env.domain_rand.randomize_p_gain_scale_lower == pytest.approx(2.9 / 3.0)
    assert cfg.env.domain_rand.randomize_p_gain_scale_upper == pytest.approx(3.1 / 3.0)
    assert cfg.env.domain_rand.randomize_d_gain_scale_lower == pytest.approx(0.9)
    assert cfg.env.domain_rand.randomize_d_gain_scale_upper == pytest.approx(1.1)
    assert cfg.env.domain_rand.randomize_friction_scale_lower == pytest.approx(0.3)
    assert cfg.env.domain_rand.randomize_friction_scale_upper == pytest.approx(3.0)
    assert cfg.env.domain_rand.scale_xml_friction_per_geom is True
    assert cfg.env.domain_rand.force_scale == pytest.approx(2.0)
    assert cfg.env.domain_rand.joint_noise_scale == pytest.approx(0.0)
    assert cfg.env.domain_rand.contact_latency == pytest.approx(0.005)
    assert cfg.env.domain_rand.contact_sensor_noise == pytest.approx(0.01)
    assert "termination_drop_distance" not in cfg.env
    assert "hora_domain_rand" not in cfg.env


def test_hora_appo_registry_isolated_from_legacy_hora_ppo() -> None:
    registered = registry.list_registered_envs()

    assert registered["LeapInhandBall0730HoraAppoRotation"]["available_backends"] == ["mujoco"]
    assert issubclass(LeapInhandBall0730HoraAppoRotationEnv, SharpaInhandRotationEnv)
    assert "LeapInhandBall0730HoraRotation" in registered


def test_hora_appo_gravity_direction_is_limited_to_three_degree_cone() -> None:
    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._cfg = LeapInhandBall0730HoraAppoRotationCfg()

    gravity = env._build_gravity_direction_randomization(4096)

    assert gravity is not None
    np.testing.assert_allclose(np.linalg.norm(gravity, axis=1), 9.81)
    tilt_deg = np.rad2deg(np.arccos(np.clip(-gravity[:, 2] / 9.81, -1.0, 1.0)))
    assert np.all(tilt_deg <= 3.0 + 1.0e-10)
    assert np.any(tilt_deg > 0.0)


def test_hora_appo_friction_scales_each_xml_geom_without_flattening_profiles() -> None:
    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._base_geom_friction = np.asarray(
        [
            [0.2, 0.005, 0.0001],
            [0.5, 0.005, 0.0001],
            [2.0, 0.05, 0.01],
            [0.2, 0.005, 0.0001],
        ],
        dtype=np.float64,
    )
    env._friction_geom_ids = {
        "metal": np.asarray([0], dtype=np.int32),
        "elastomer": np.asarray([1], dtype=np.int32),
        "object": np.asarray([2], dtype=np.int32),
    }
    env._cfg = SimpleNamespace(
        domain_rand=SimpleNamespace(scale_xml_friction_per_geom=True)
    )

    result = env._build_friction_randomization(
        2,
        np.asarray([[1.0], [0.5]], dtype=np.float64),
    )

    assert result is not None
    np.testing.assert_allclose(result[0], env._base_geom_friction)
    np.testing.assert_allclose(result[1, :3], env._base_geom_friction[:3] * 0.5)
    np.testing.assert_allclose(result[1, 3], env._base_geom_friction[3])


def test_leap_embodiment_dimensions_and_nominal_anchor() -> None:
    cfg = LeapInhandBall0730HoraAppoRotationCfg()

    assert cfg.num_hand_dofs == 16
    assert len(cfg.fingertip_body_names) == 4
    assert len(cfg.sensor.tactile_force_sensor_names) == 4
    assert cfg.tactile_diagnostic_names == ["index", "middle", "ring", "thumb"]
    frame_dim = 2 * cfg.num_hand_dofs + len(cfg.sensor.tactile_force_sensor_names)
    assert frame_dim == 36
    assert cfg.obs_lag_steps * frame_dim == 108
    assert cfg.obs_lag_steps * frame_dim + cfg.critic_info_dim == 117
    assert (cfg.prop_hist_len, frame_dim) == (30, 36)
    np.testing.assert_allclose(
        cfg.default_object_pose[:3],
        [-0.032440416893199604, 0.041151239943936, 0.664301098275159],
    )
    assert len(cfg.default_hand_joint_pos) == 16
    assert cfg.reset_height_upper - cfg.reset_height_lower == pytest.approx(0.06)


def test_sharpa_defaults_remain_22_dof_and_model_owned_object_anchor() -> None:
    cfg = SharpaInhandBaseCfg()

    assert cfg.num_hand_dofs == 22
    assert len(cfg.default_hand_joint_pos) == 22
    assert cfg.default_object_pose == []


def test_target_update_matches_sharpa_math() -> None:
    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._num_envs = 2
    env._num_action = 16
    env._np_dtype = np.float32
    env.default_angles = np.zeros(16, dtype=np.float32)
    env._target_lower = np.full(16, -0.9, dtype=np.float32)
    env._target_upper = np.full(16, 0.9, dtype=np.float32)
    env._cfg = SimpleNamespace(
        clip_actions=1.0,
        control_config=SimpleNamespace(action_scale=1.0 / 24.0),
    )
    previous = np.stack([np.linspace(-0.8, 0.8, 16), np.linspace(0.8, -0.8, 16)]).astype(np.float32)
    action = np.stack([np.full(16, 2.0), np.full(16, -2.0)]).astype(np.float32)
    state = SimpleNamespace(info={"prev_targets": previous.copy()})

    targets = SharpaInhandBaseEnv.apply_action(env, action, state)
    expected = np.clip(previous + np.clip(action, -1.0, 1.0) / 24.0, -0.9, 0.9)

    np.testing.assert_allclose(targets, expected, rtol=0.0, atol=1.0e-7)
    np.testing.assert_array_equal(state.info["current_actions"], np.clip(action, -1.0, 1.0))
    np.testing.assert_array_equal(state.info["last_actions"], np.clip(action, -1.0, 1.0))


def test_reward_matches_sharpa_formula() -> None:
    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._num_envs = 2
    env._np_dtype = np.float64
    env._enable_reward_log = False
    env._rot_axis = np.asarray([0.0, 0.0, 1.0])
    env.default_angles = np.zeros(16)
    env._cfg = SimpleNamespace(ctrl_dt=0.05)
    env._reward_cfg = RewardConfig()
    dof_pos = np.full((2, 16), 0.1)
    dof_vel = np.full((2, 16), 0.2)
    object_pos = np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    object_linvel = np.asarray([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    object_angvel = np.asarray([[0.0, 0.0, 0.8], [0.0, 0.0, -0.8]])
    torques = np.full((2, 16), 0.25)
    anchor = np.zeros((2, 3))
    info = {"object_pos_anchor": anchor, "steps": np.zeros(2, dtype=np.uint32)}

    reward = env._compute_reward(
        info,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        object_pos=object_pos,
        object_linvel=object_linvel,
        object_angvel=object_angvel,
        torques=torques,
    )
    expected = 0.05 * (
        2.5 * np.asarray([0.5, -0.5])
        - 0.3 * np.sum(np.abs(object_linvel), axis=1)
        - 0.4 * np.sum(np.square(dof_pos), axis=1)
        - 0.1 * np.sum(np.square(torques), axis=1)
        - 0.5 * np.square(np.sum(torques * dof_vel, axis=1))
        + 0.003 / (np.linalg.norm(object_pos - anchor, axis=1) + 0.001)
    )

    np.testing.assert_allclose(reward, expected, rtol=0.0, atol=1.0e-12)


def test_force_sensor_contract_is_not_binary_found_sensor() -> None:
    scene = SCENE.read_text(encoding="utf-8")

    for sensor_name in LEAP_TACTILE_FORCE_SENSOR_NAMES:
        assert f'name="{sensor_name}"' in scene
    assert scene.count('data="force"') >= 4
    assert scene.count('reduce="netforce"') >= 4


def test_mujoco_model_exposes_four_vector_force_sensors() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))

    sensor_dims = {}
    for sensor_id in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        if name in LEAP_TACTILE_FORCE_SENSOR_NAMES:
            sensor_dims[name] = int(model.sensor_dim[sensor_id])
    assert sensor_dims == {name: 3 for name in LEAP_TACTILE_FORCE_SENSOR_NAMES}


def test_tactile_force_read_fails_closed_for_missing_sensor() -> None:
    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._num_envs = 1
    env._num_tactile = 4
    env._np_dtype = np.float32
    env._cfg = SimpleNamespace(
        sensor=SimpleNamespace(tactile_force_sensor_names=list(LEAP_TACTILE_FORCE_SENSOR_NAMES))
    )
    env._backend = SimpleNamespace(
        get_sensor_data=lambda name: (_ for _ in ()).throw(KeyError(name))
    )

    with pytest.raises(KeyError, match="leap_index_tactile_force"):
        env._read_tactile_force()


def test_scale_cache_loading_fails_closed_without_all_buckets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora_appo as module

    monkeypatch.setattr(module, "resolve_grasp_cache_files", lambda path: path)
    env = SimpleNamespace(
        _grasp_cache=None,
        _num_action=16,
        scale_values=np.asarray([0.8, 1.0]),
        cfg=SimpleNamespace(grasp_cache_path=str(tmp_path / "leap_scale")),
    )
    cache = np.zeros((2, 23), dtype=np.float32)
    np.save(tmp_path / "leap_scale_1.npy", cache)

    with pytest.raises(RuntimeError, match="leap_scale_0.8.npy"):
        LeapInhandBall0730HoraAppoDRProvider()._load_grasp_cache(env)


def test_scale_cache_error_includes_generation_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora_appo as module

    monkeypatch.setattr(module, "resolve_grasp_cache_files", lambda path: path)
    env = SimpleNamespace(
        _grasp_cache=None,
        _num_action=16,
        scale_values=np.asarray([1.0]),
        cfg=SimpleNamespace(grasp_cache_path=str(tmp_path / "missing")),
    )

    with pytest.raises(RuntimeError, match="leap_collect_hora_grasps.sh"):
        LeapInhandBall0730HoraAppoDRProvider()._load_grasp_cache(env)


def test_real_mujoco_reset_and_step_contract(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    scales = [0.8, 1.0]
    cache_prefix = tmp_path / "leap_hora"
    nominal_cfg = LeapInhandBall0730HoraAppoRotationCfg()
    row = np.asarray(
        [*nominal_cfg.default_hand_joint_pos, *nominal_cfg.default_object_pose],
        dtype=np.float64,
    )
    for scale in scales:
        np.save(
            resolve_grasp_cache_file(str(cache_prefix), scale),
            np.broadcast_to(row, (2, 23)).copy(),
        )

    domain_rand = SharpaDomainRandConfig(
        scale_list=scales,
        randomize_gravity_direction=False,
        randomize_pd_gains=False,
        randomize_friction=False,
        randomize_com=False,
        randomize_mass=False,
        force_scale=0.0,
        joint_noise_scale=0.0,
        contact_latency=0.0,
        contact_sensor_noise=0.0,
    )
    cfg = LeapInhandBall0730HoraAppoRotationCfg(
        grasp_cache_path=str(cache_prefix),
        domain_rand=domain_rand,
        reward_config=RewardConfig(),
    )
    env = LeapInhandBall0730HoraAppoRotationEnv(cfg, num_envs=2, backend_type="mujoco")
    try:
        variant_0 = env._backend.get_playback_model_variant_spec(0)
        variant_1 = env._backend.get_playback_model_variant_spec(1)
        assert variant_0 is not None
        assert variant_1 is not None
        assert variant_0.mesh_scale_multipliers[0].mesh_name == "leap_ball_visual_mesh"
        np.testing.assert_allclose(variant_0.mesh_scale_multipliers[0].multiplier, [0.8] * 3)
        np.testing.assert_allclose(variant_1.mesh_scale_multipliers[0].multiplier, [1.0] * 3)

        from unilab.base.backend.mujoco.playback import resolve_render_play_model_files

        viewer_dir = tmp_path / "viewer_models"
        viewer_dir.mkdir()
        viewer_files = resolve_render_play_model_files(env, num_envs=2, tmp_dir=viewer_dir)
        assert isinstance(viewer_files, list)
        viewer_models = [mujoco.MjModel.from_binary_path(path) for path in viewer_files]
        mesh_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "leap_ball_visual_mesh")
            for model in viewer_models
        ]
        assert all(mesh_id >= 0 for mesh_id in mesh_ids)
        np.testing.assert_allclose(viewer_models[0].mesh_scale[mesh_ids[0]], [0.0335 * 0.8] * 3)
        np.testing.assert_allclose(viewer_models[1].mesh_scale[mesh_ids[1]], [0.0335] * 3)
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 108)
        assert state.obs["critic"].shape == (2, 117)
        assert state.info["proprio_hist"].shape == (2, 30, 36)
        assert state.info["critic_info"].shape == (2, 9)
        state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        assert set(state.info["log"]) >= {
            "diagnostic/contact_ratio/index",
            "diagnostic/contact_ratio/middle",
            "diagnostic/contact_ratio/ring",
            "diagnostic/contact_ratio/thumb",
        }
        np.testing.assert_allclose(
            state.info["reset_height_lower"],
            row[18] - 0.5 * (cfg.reset_height_upper - cfg.reset_height_lower),
        )
        np.testing.assert_allclose(
            state.info["reset_height_upper"],
            row[18] + 0.5 * (cfg.reset_height_upper - cfg.reset_height_lower),
        )
    finally:
        env.close()
