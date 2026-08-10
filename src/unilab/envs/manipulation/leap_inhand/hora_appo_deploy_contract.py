"""Pure no-tactile deployment contract for the LEAP HORA APPO student.

Inputs to this module must already use the MuJoCo/training joint convention.
Real motor order, sign, and offset calibration are deliberately outside this
contract; a deployment without a separately validated calibration must fail
closed before calling these builders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "leap_hora_appo_no_tactile_v1"
NUM_JOINTS = 16
FRAME_DIM = 32
ACTOR_HISTORY_LEN = 3
ACTOR_OBS_DIM = 96
PROPRIO_HISTORY_LEN = 30
PROPRIO_FRAME_DIM = 32
PRIV_INFO_DIM = 9
ACTION_DIM = 16
CONTROL_DT = 0.05
ACTION_SCALE = 1.0 / 24.0

_NORMALIZATION_EPSILON = 1.0e-8


def _finite_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be a real numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _joint_array(value: object, *, name: str) -> np.ndarray:
    array = _finite_array(value, name=name)
    if array.ndim < 1 or array.shape[-1] != NUM_JOINTS:
        raise ValueError(f"{name} must have shape (..., {NUM_JOINTS})")
    return array


def _bounds(lower: object, upper: object) -> tuple[np.ndarray, np.ndarray]:
    lower_array = _finite_array(lower, name="lower")
    upper_array = _finite_array(upper, name="upper")
    if lower_array.shape != (NUM_JOINTS,) or upper_array.shape != (NUM_JOINTS,):
        raise ValueError(f"lower and upper must both have shape ({NUM_JOINTS},)")
    if np.any(upper_array <= lower_array):
        raise ValueError("upper must be greater than lower for every joint")
    return lower_array, upper_array


def normalize_joint_position(
    measured_q: object,
    lower: object,
    upper: object,
) -> np.ndarray:
    """Apply the exact Sharpa training normalization to measured joint positions."""

    measured = _joint_array(measured_q, name="measured_q")
    lower_array, upper_array = _bounds(lower, upper)
    dtype = np.dtype(
        np.result_type(measured.dtype, lower_array.dtype, upper_array.dtype, np.float32)
    )
    measured = np.asarray(measured, dtype=dtype)
    lower_array = np.asarray(lower_array, dtype=dtype)
    upper_array = np.asarray(upper_array, dtype=dtype)
    return np.asarray(
        (2.0 * measured - upper_array - lower_array)
        / (upper_array - lower_array + _NORMALIZATION_EPSILON),
        dtype=dtype,
    )


def build_policy_frame(
    measured_q: object,
    previous_commanded_target: object,
    lower: object,
    upper: object,
) -> np.ndarray:
    """Build ``[normalized measured q16, previous commanded target16]``.

    The target channels intentionally retain their training semantics: raw
    commanded joint targets in radians. Tactile is not accepted by this API.
    """

    q_norm = normalize_joint_position(measured_q, lower, upper)
    target = _joint_array(previous_commanded_target, name="previous_commanded_target")
    if q_norm.shape != target.shape:
        raise ValueError("measured_q and previous_commanded_target must have the same shape")
    dtype = np.dtype(np.result_type(q_norm.dtype, target.dtype, np.float32))
    frame = np.concatenate(
        [np.asarray(q_norm, dtype=dtype), np.asarray(target, dtype=dtype)], axis=-1
    )
    if frame.shape[-1] != FRAME_DIM:  # pragma: no cover - constant invariant
        raise RuntimeError(f"policy frame width must be {FRAME_DIM}")
    return np.asarray(frame, dtype=dtype)


def integrate_action(
    previous_target: object,
    action: object,
    target_lower: object,
    target_upper: object,
) -> np.ndarray:
    """Match Sharpa incremental target integration and clipping exactly."""

    previous = _joint_array(previous_target, name="previous_target")
    action_array = _joint_array(action, name="action")
    lower, upper = _bounds(target_lower, target_upper)
    if previous.shape != action_array.shape:
        raise ValueError("previous_target and action must have the same shape")
    dtype = np.dtype(
        np.result_type(previous.dtype, action_array.dtype, lower.dtype, upper.dtype, np.float32)
    )
    target = np.asarray(previous, dtype=dtype) + ACTION_SCALE * np.clip(
        np.asarray(action_array, dtype=dtype), -1.0, 1.0
    )
    return np.asarray(np.clip(target, lower, upper), dtype=dtype)


class HistoryBuffer:
    """Single-environment, oldest-first fixed-length frame history."""

    def __init__(self, history_len: int, frame_dim: int = FRAME_DIM) -> None:
        if history_len <= 0 or frame_dim <= 0:
            raise ValueError("history_len and frame_dim must be positive")
        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self._buffer: np.ndarray | None = None

    def _frame(self, frame: object) -> np.ndarray:
        array = _finite_array(frame, name="frame")
        if array.shape != (self.frame_dim,):
            raise ValueError(f"frame must have shape ({self.frame_dim},)")
        return array

    def reset(self, frame: object) -> None:
        validated = self._frame(frame)
        self._buffer = np.repeat(validated[None, :], self.history_len, axis=0)

    def push(self, frame: object) -> None:
        if self._buffer is None:
            raise RuntimeError("history must be reset before push")
        validated = self._frame(frame)
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = validated

    def last_n(self, count: int) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("history must be reset before it can be read")
        if count <= 0 or count > self.history_len:
            raise ValueError(f"count must be in [1, {self.history_len}]")
        return self._buffer[-count:].copy()

    def flatten_oldest_first(self) -> np.ndarray:
        return self.last_n(self.history_len).reshape(-1).copy()


@dataclass(frozen=True)
class StudentDeploymentObservation:
    actor_obs: np.ndarray
    proprio_hist: np.ndarray


class StudentDeploymentObservationBuilder:
    """Build the only real-time neural inputs required by the distilled student."""

    def __init__(self, lower: object, upper: object) -> None:
        lower_array, upper_array = _bounds(lower, upper)
        self.lower = lower_array.copy()
        self.upper = upper_array.copy()
        self._actor_history = HistoryBuffer(ACTOR_HISTORY_LEN)
        self._proprio_history = HistoryBuffer(PROPRIO_HISTORY_LEN)

    def _observation(self) -> StudentDeploymentObservation:
        actor_obs = self._actor_history.flatten_oldest_first()
        proprio_hist = self._proprio_history.last_n(PROPRIO_HISTORY_LEN)
        return StudentDeploymentObservation(actor_obs=actor_obs, proprio_hist=proprio_hist)

    def reset(self, measured_q: object, current_target: object) -> StudentDeploymentObservation:
        """Initialize both histories; startup must pass ``current_target=measured_q``."""

        frame = build_policy_frame(measured_q, current_target, self.lower, self.upper)
        if frame.shape != (FRAME_DIM,):
            raise ValueError("deployment builder accepts one environment at a time")
        self._actor_history.reset(frame)
        self._proprio_history.reset(frame)
        return self._observation()

    def step(self, measured_q: object, current_target: object) -> StudentDeploymentObservation:
        frame = build_policy_frame(measured_q, current_target, self.lower, self.upper)
        if frame.shape != (FRAME_DIM,):
            raise ValueError("deployment builder accepts one environment at a time")
        self._actor_history.push(frame)
        self._proprio_history.push(frame)
        return self._observation()


def build_deploy_contract_manifest(
    *,
    joint_names: object,
    joint_lower: object,
    joint_upper: object,
) -> dict[str, Any]:
    """Build a machine-readable contract using model/export-owned joint metadata."""

    names = [str(name) for name in joint_names]  # type: ignore[union-attr]
    if len(names) != NUM_JOINTS or len(set(names)) != NUM_JOINTS:
        raise ValueError(f"joint_names must contain {NUM_JOINTS} unique names")
    lower, upper = _bounds(joint_lower, joint_upper)
    return {
        "contract_version": CONTRACT_VERSION,
        "num_joints": NUM_JOINTS,
        "frame_components": [
            {"name": "joint_position_normalized", "dim": NUM_JOINTS},
            {"name": "previous_commanded_target", "dim": NUM_JOINTS},
        ],
        "frame_dim": FRAME_DIM,
        "actor_history_len": ACTOR_HISTORY_LEN,
        "actor_obs_dim": ACTOR_OBS_DIM,
        "proprio_history_len": PROPRIO_HISTORY_LEN,
        "proprio_frame_dim": PROPRIO_FRAME_DIM,
        "privileged_info_dim": PRIV_INFO_DIM,
        "action_dim": ACTION_DIM,
        "control_dt": CONTROL_DT,
        "action_scale": ACTION_SCALE,
        "requires_tactile": False,
        "joint_names": names,
        "joint_lower": lower.tolist(),
        "joint_upper": upper.tolist(),
        "real_motor_calibration_included": False,
    }


def write_deploy_contract_manifest(
    path: str | Path,
    *,
    joint_names: object,
    joint_lower: object,
    joint_upper: object,
) -> Path:
    """Write ``leap_hora_deploy_contract.json`` without inventing motor mapping."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_deploy_contract_manifest(
        joint_names=joint_names,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


__all__ = [
    "ACTION_DIM",
    "ACTION_SCALE",
    "ACTOR_HISTORY_LEN",
    "ACTOR_OBS_DIM",
    "CONTRACT_VERSION",
    "CONTROL_DT",
    "FRAME_DIM",
    "HistoryBuffer",
    "NUM_JOINTS",
    "PRIV_INFO_DIM",
    "PROPRIO_FRAME_DIM",
    "PROPRIO_HISTORY_LEN",
    "StudentDeploymentObservation",
    "StudentDeploymentObservationBuilder",
    "build_deploy_contract_manifest",
    "build_policy_frame",
    "integrate_action",
    "normalize_joint_position",
    "write_deploy_contract_manifest",
]
