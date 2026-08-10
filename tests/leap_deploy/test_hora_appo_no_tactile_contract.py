from __future__ import annotations

import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.hora.distill import load_distilled_checkpoint
from unilab.algos.torch.hora.models import HoraActorModel, HoraSharedActorCritic, ProprioAdaptTConv
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora_appo import (
    LeapInhandBall0730HoraAppoRotationEnv,
)
from unilab.envs.manipulation.leap_inhand.hora_appo_deploy_contract import (
    ACTION_DIM,
    ACTION_SCALE,
    ACTOR_HISTORY_LEN,
    ACTOR_OBS_DIM,
    CONTROL_DT,
    FRAME_DIM,
    NUM_JOINTS,
    PRIV_INFO_DIM,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LEN,
    HistoryBuffer,
    StudentDeploymentObservationBuilder,
    build_deploy_contract_manifest,
    build_policy_frame,
    integrate_action,
    normalize_joint_position,
)
from unilab.envs.manipulation.sharpa_inhand.base import SharpaInhandBaseEnv
from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationEnv

ROOT = Path(__file__).resolve().parents[2]


def _training_frame_env(
    lower: np.ndarray,
    upper: np.ndarray,
) -> LeapInhandBall0730HoraAppoRotationEnv:
    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._num_action = NUM_JOINTS
    env._num_tactile = 4
    env._np_dtype = np.float32
    env._ctrl_lower = np.asarray(lower, dtype=np.float32)
    env._ctrl_upper = np.asarray(upper, dtype=np.float32)
    env._observation_mode = "separated"
    env._cfg = SimpleNamespace(
        obs=SimpleNamespace(enable_tactile=False, enable_contact_pos=False),
        domain_rand=SimpleNamespace(joint_noise_scale=0.0),
        clip_obs=5.0,
        obs_history_len=80,
        obs_lag_steps=ACTOR_HISTORY_LEN,
        prop_hist_len=PROPRIO_HISTORY_LEN,
        critic_info_dim=PRIV_INFO_DIM,
    )

    def zero_critic_info(self, info, batch_size, object_pos):
        del self, object_pos
        critic_info = np.zeros((batch_size, PRIV_INFO_DIM), dtype=np.float32)
        info["critic_info"] = critic_info
        return critic_info

    env._build_critic_info = MethodType(zero_critic_info, env)
    return env


def _student_actor(seed: int = 7) -> HoraActorModel:
    torch.manual_seed(seed)
    shared = HoraSharedActorCritic(
        obs_dim=ACTOR_OBS_DIM,
        action_dim=ACTION_DIM,
        priv_info_dim=PRIV_INFO_DIM,
        priv_info_embed_dim=PRIV_INFO_DIM,
        actor_hidden_dims=(64, 32),
        priv_mlp_hidden_dims=(32, PRIV_INFO_DIM),
        obs_normalization=True,
        use_student_encoder=True,
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        proprio_frame_dim=PROPRIO_FRAME_DIM,
    )
    example = TensorDict(
        {
            "actor": torch.zeros(2, ACTOR_OBS_DIM),
            "proprio_hist": torch.zeros(2, PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM),
        },
        batch_size=2,
    )
    return HoraActorModel(
        example,
        {"actor": ["actor"]},
        "actor",
        ACTION_DIM,
        shared_model=shared,
        use_student_encoder=True,
    )


def test_fixed_no_tactile_contract_and_api() -> None:
    assert NUM_JOINTS == 16
    assert FRAME_DIM == 32
    assert ACTOR_HISTORY_LEN == 3
    assert ACTOR_OBS_DIM == 96
    assert PROPRIO_HISTORY_LEN == 30
    assert PROPRIO_FRAME_DIM == 32
    assert PRIV_INFO_DIM == 9
    assert ACTION_DIM == 16
    assert CONTROL_DT == pytest.approx(0.05)
    assert ACTION_SCALE == pytest.approx(1.0 / 24.0)
    assert "tactile" not in inspect.signature(build_policy_frame).parameters
    assert "tactile" not in inspect.signature(StudentDeploymentObservationBuilder.step).parameters


def test_policy_frame_exact_layout_and_tactile_independence() -> None:
    lower = np.linspace(-1.5, -0.5, NUM_JOINTS, dtype=np.float32)
    upper = np.linspace(0.5, 1.5, NUM_JOINTS, dtype=np.float32)
    measured_q = lower + np.float32(0.3) * (upper - lower)
    previous_target = np.linspace(-0.4, 0.4, NUM_JOINTS, dtype=np.float32)
    env = _training_frame_env(lower, upper)

    frame_a = env._build_policy_frame(
        measured_q[None],
        previous_target[None],
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 12), dtype=np.float32),
        add_noise=False,
    )
    frame_b = env._build_policy_frame(
        measured_q[None],
        previous_target[None],
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        np.ones((1, 12), dtype=np.float32),
        add_noise=False,
    )
    deploy_frame = build_policy_frame(measured_q, previous_target, lower, upper)

    assert frame_a.shape == (1, FRAME_DIM)
    np.testing.assert_array_equal(frame_a, frame_b)
    np.testing.assert_allclose(frame_a[0], deploy_frame, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        frame_a[0, :NUM_JOINTS],
        normalize_joint_position(measured_q, lower, upper),
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(frame_a[0, NUM_JOINTS:], previous_target)


def test_training_and_deployment_histories_match_for_thirty_steps() -> None:
    rng = np.random.default_rng(19)
    lower = np.full(NUM_JOINTS, -1.5, dtype=np.float32)
    upper = np.full(NUM_JOINTS, 1.5, dtype=np.float32)
    env = _training_frame_env(lower, upper)
    builder = StudentDeploymentObservationBuilder(lower, upper)
    info: dict[str, object] = {}

    latest = None
    for step in range(PROPRIO_HISTORY_LEN):
        measured = rng.uniform(-1.0, 1.0, NUM_JOINTS).astype(np.float32)
        target = rng.uniform(-0.8, 0.8, NUM_JOINTS).astype(np.float32)
        info["prev_targets"] = target[None]
        tactile = rng.uniform(0.0, 4.0, (1, 4)).astype(np.float32)
        training_obs = env._compute_obs_from_inputs(
            info,
            measured[None],
            np.zeros((1, 3), dtype=np.float32),
            tactile,
            np.zeros((1, 12), dtype=np.float32),
        )
        latest = builder.reset(measured, target) if step == 0 else builder.step(measured, target)
        np.testing.assert_allclose(training_obs["obs"][0], latest.actor_obs, atol=1.0e-7)
        np.testing.assert_allclose(
            np.asarray(info["proprio_hist"])[0], latest.proprio_hist, atol=1.0e-7
        )

    assert latest is not None
    assert latest.actor_obs.shape == (ACTOR_OBS_DIM,)
    assert latest.proprio_hist.shape == (PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM)


def test_tactile_cannot_change_actor_proprio_or_critic_policy_history() -> None:
    lower = np.full(NUM_JOINTS, -1.0, dtype=np.float32)
    upper = np.full(NUM_JOINTS, 1.0, dtype=np.float32)
    measured = np.linspace(-0.5, 0.5, NUM_JOINTS, dtype=np.float32)[None]
    target = np.linspace(0.3, -0.3, NUM_JOINTS, dtype=np.float32)[None]
    outputs = []
    for tactile in (
        np.zeros((1, 4), dtype=np.float32),
        np.full((1, 4), 4.0, dtype=np.float32),
    ):
        env = _training_frame_env(lower, upper)
        info: dict[str, object] = {"prev_targets": target.copy()}
        obs = env._compute_obs_from_inputs(
            info,
            measured,
            np.zeros((1, 3), dtype=np.float32),
            tactile,
            np.zeros((1, 12), dtype=np.float32),
        )
        outputs.append((obs, np.asarray(info["proprio_hist"])))

    np.testing.assert_array_equal(outputs[0][0]["obs"], outputs[1][0]["obs"])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(
        outputs[0][0]["critic"][:, :ACTOR_OBS_DIM],
        outputs[1][0]["critic"][:, :ACTOR_OBS_DIM],
    )
    np.testing.assert_array_equal(outputs[0][0]["critic"], outputs[1][0]["critic"])


def test_history_order_and_action_integration_match_training() -> None:
    history = HistoryBuffer(3, 2)
    history.reset(np.asarray([0.0, 1.0], dtype=np.float32))
    history.push(np.asarray([2.0, 3.0], dtype=np.float32))
    history.push(np.asarray([4.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(history.flatten_oldest_first(), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(history.last_n(2), [[2, 3], [4, 5]])

    env = object.__new__(LeapInhandBall0730HoraAppoRotationEnv)
    env._num_envs = 1
    env._num_action = NUM_JOINTS
    env._np_dtype = np.float32
    env.default_angles = np.zeros(NUM_JOINTS, dtype=np.float32)
    env._target_lower = np.full(NUM_JOINTS, -0.9, dtype=np.float32)
    env._target_upper = np.full(NUM_JOINTS, 0.9, dtype=np.float32)
    env._cfg = SimpleNamespace(
        clip_actions=1.0,
        control_config=SimpleNamespace(action_scale=ACTION_SCALE),
    )
    previous = np.linspace(-0.8, 0.8, NUM_JOINTS, dtype=np.float32)[None]
    action = np.linspace(-2.0, 2.0, NUM_JOINTS, dtype=np.float32)[None]
    state = SimpleNamespace(info={"prev_targets": previous.copy()})
    training_target = SharpaInhandBaseEnv.apply_action(env, action, state)
    deploy_target = integrate_action(previous[0], action[0], env._target_lower, env._target_upper)
    np.testing.assert_allclose(training_target[0], deploy_target, rtol=0.0, atol=1.0e-7)


def test_leap_distillation_owner_composes_appo_teacher() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "hora_distill"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=["task=leap_inhand_ball_0730/mujoco"],
        )
    assert cfg.teacher.algo_family == "appo"
    assert cfg.teacher.task == "leap_inhand_ball_0730/mujoco_hora"
    assert cfg.training.task_name == "LeapInhandBall0730HoraAppoRotation"
    assert cfg.algo.algo_log_name == "hora_distill"


def test_adaptation_forward_backward_and_student_inference_need_no_privileged_inputs() -> None:
    torch.manual_seed(31)
    encoder = ProprioAdaptTConv(frame_dim=PROPRIO_FRAME_DIM, latent_dim=PRIV_INFO_DIM)
    history = torch.randn(4, PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM)
    target = torch.randn(4, PRIV_INFO_DIM)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1.0e-3)
    before = encoder.low_dim_proj.weight.detach().clone()
    latent = encoder(history)
    assert latent.shape == (4, PRIV_INFO_DIM)
    loss = torch.mean((latent - target) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, encoder.low_dim_proj.weight)

    actor = _student_actor()
    student_inputs = TensorDict(
        {
            "actor": torch.randn(4, ACTOR_OBS_DIM),
            "proprio_hist": torch.randn(4, PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM),
        },
        batch_size=4,
    )
    with torch.inference_mode():
        actions = actor(student_inputs, stochastic_output=False)
    assert actions.shape == (4, ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_student_checkpoint_and_both_normalizers_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(43)
    actor = _student_actor(seed=43)
    hist_normalizer = EmpiricalNormalization((PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM), device="cpu")
    actor_obs = torch.randn(5, ACTOR_OBS_DIM)
    raw_history = torch.randn(5, PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM)
    actor.shared.obs_normalizer.train()
    actor.shared.obs_normalizer(actor_obs)
    actor.shared.obs_normalizer.eval()
    hist_normalizer.train()
    hist_normalizer(raw_history)
    hist_normalizer.eval()
    normalized_history = hist_normalizer(raw_history, update=False)
    inputs = TensorDict(
        {"actor": actor_obs, "proprio_hist": normalized_history},
        batch_size=5,
    )
    with torch.inference_mode():
        expected = actor(inputs, stochastic_output=False).clone()

    checkpoint = tmp_path / "hora_stage2_1.pt"
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "history_normalizer": hist_normalizer.state_dict(),
        },
        checkpoint,
    )
    restored_actor = _student_actor(seed=99)
    restored_normalizer = EmpiricalNormalization(
        (PROPRIO_HISTORY_LEN, PROPRIO_FRAME_DIM), device="cpu"
    )
    load_distilled_checkpoint(
        restored_actor,
        restored_normalizer,
        checkpoint,
        device=torch.device("cpu"),
    )
    restored_actor.eval()
    restored_normalizer.eval()
    restored_inputs = TensorDict(
        {
            "actor": actor_obs,
            "proprio_hist": restored_normalizer(raw_history, update=False),
        },
        batch_size=5,
    )
    with torch.inference_mode():
        actual = restored_actor(restored_inputs, stochastic_output=False)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(restored_normalizer.mean, hist_normalizer.mean)
    torch.testing.assert_close(
        restored_actor.shared.obs_normalizer.mean,
        actor.shared.obs_normalizer.mean,
    )


def test_old_108d_teacher_fails_closed_with_diagnostic() -> None:
    from unilab.algos.torch.hora.appo_runner import (
        _validate_hora_actor_observation_contract,
    )

    checkpoint = {
        "actor": {
            "shared.trunk.net.0.weight": torch.zeros(32, 108 + PRIV_INFO_DIM),
        }
    }
    with pytest.raises(ValueError, match="(?s)108 dims.*96 dims.*simulated tactile"):
        _validate_hora_actor_observation_contract(
            checkpoint,
            current_actor_obs_dim=ACTOR_OBS_DIM,
            current_privileged_latent_dim=PRIV_INFO_DIM,
        )


def test_reward_has_no_tactile_or_contact_gate() -> None:
    source = inspect.getsource(SharpaInhandRotationEnv._compute_reward)
    signature = inspect.signature(SharpaInhandRotationEnv._compute_reward)
    assert "tactile" not in signature.parameters
    assert "contact" not in signature.parameters
    assert "tactile" not in source
    assert "contact" not in source


def test_manifest_records_model_metadata_and_no_tactile_requirement() -> None:
    names = [f"joint_{index}" for index in range(NUM_JOINTS)]
    lower = np.linspace(-1.5, -0.5, NUM_JOINTS)
    upper = np.linspace(0.5, 1.5, NUM_JOINTS)
    manifest = build_deploy_contract_manifest(
        joint_names=names,
        joint_lower=lower,
        joint_upper=upper,
    )
    assert manifest["contract_version"] == "leap_hora_appo_no_tactile_v1"
    assert manifest["frame_dim"] == FRAME_DIM
    assert manifest["actor_obs_dim"] == ACTOR_OBS_DIM
    assert manifest["requires_tactile"] is False
    assert manifest["real_motor_calibration_included"] is False
    assert manifest["joint_names"] == names
    np.testing.assert_array_equal(manifest["joint_lower"], lower)
    np.testing.assert_array_equal(manifest["joint_upper"], upper)
