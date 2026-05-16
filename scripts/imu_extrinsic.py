#!/usr/bin/env python3
"""IMU -> platform body (mocap world) fixed extrinsic rotation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

DEFAULT_EXTRINSIC_FILENAME = "cdpr_imu_extrinsic.json"


def default_extrinsic_path(base_dir: Optional[Path] = None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    return base_dir / DEFAULT_EXTRINSIC_FILENAME


def normalize_quat(quat_xyzw: np.ndarray) -> Optional[np.ndarray]:
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        return None
    return q / np.linalg.norm(q)


@dataclass
class ImuExtrinsic:
    """Right-multiply on IMU attitude: R_world_body = R_world_imu @ R_imu_to_body."""

    quat_imu_to_body: np.ndarray  # xyzw
    n_samples: int = 0
    residual_angle_deg_rms: float = 0.0

    @property
    def r_imu_to_body(self) -> R:
        return R.from_quat(self.quat_imu_to_body)

    def apply_quat(self, quat_imu_xyzw: np.ndarray) -> Optional[np.ndarray]:
        q = normalize_quat(quat_imu_xyzw)
        if q is None:
            return None
        return (R.from_quat(q) * self.r_imu_to_body).as_quat()

    def apply_vector(self, vec_imu: np.ndarray) -> np.ndarray:
        """Map free vector from IMU sensor frame to platform body frame."""
        return self.r_imu_to_body.inv().apply(np.asarray(vec_imu, dtype=float).reshape(3))

    def to_dict(self) -> dict:
        r = self.r_imu_to_body
        yaw, pitch, roll = r.as_euler("ZYX", degrees=False)
        return {
            "description": (
                "Fixed rotation IMU -> CDPR platform body (aligned with mocap rigid body). "
                "R_world_body = R_world_imu @ R_imu_to_body."
            ),
            "convention": "scipy: q_body = q_imu * q_imu_to_body",
            "n_samples": int(self.n_samples),
            "residual_angle_deg_rms": float(self.residual_angle_deg_rms),
            "quat_imu_to_body": [float(x) for x in self.quat_imu_to_body],
            "R_imu_to_body": r.as_matrix().tolist(),
            "rpy_imu_to_body_deg": [
                float(np.rad2deg(roll)),
                float(np.rad2deg(pitch)),
                float(np.rad2deg(yaw)),
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImuExtrinsic":
        if "quat_imu_to_body" in data:
            q = normalize_quat(np.asarray(data["quat_imu_to_body"], dtype=float))
        elif "R_imu_to_body" in data:
            q = R.from_matrix(np.asarray(data["R_imu_to_body"], dtype=float)).as_quat()
        else:
            raise KeyError("JSON must contain quat_imu_to_body or R_imu_to_body")
        if q is None:
            raise ValueError("Invalid quaternion in extrinsic JSON")
        return cls(
            quat_imu_to_body=q,
            n_samples=int(data.get("n_samples", 0)),
            residual_angle_deg_rms=float(data.get("residual_angle_deg_rms", 0.0)),
        )


def load_imu_extrinsic(
    path: Union[str, Path],
    *,
    required: bool = False,
) -> Optional[ImuExtrinsic]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if not p.is_file():
        if required:
            raise FileNotFoundError(f"IMU extrinsic file not found: {p}")
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ext = ImuExtrinsic.from_dict(data)
    return ext


def save_imu_extrinsic(ext: ImuExtrinsic, path: Union[str, Path]) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(ext.to_dict(), f, indent=2)
        f.write("\n")
    return p


def nearest_indices(t_query: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    if t_query.size == 0 or t_ref.size == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(t_ref, t_query)
    idx = np.clip(idx, 0, len(t_ref) - 1)
    idx0 = np.clip(idx - 1, 0, len(t_ref) - 1)
    left = np.abs(t_ref[idx0] - t_query)
    right = np.abs(t_ref[idx] - t_query)
    return np.where(left <= right, idx0, idx)


def estimate_imu_extrinsic(
    quats_imu: Sequence[np.ndarray],
    quats_body: Sequence[np.ndarray],
) -> ImuExtrinsic:
    """
    Estimate R_imu_to_body from paired orientations in the same world frame.

    R_world_body = R_world_imu @ R_imu_to_body  =>  R_imu_to_body = R_world_imu^{-1} @ R_world_body
    """
    if len(quats_imu) < 3:
        raise ValueError(f"Need >= 3 pose pairs, got {len(quats_imu)}")
    rel: List[R] = []
    for qi, qb in zip(quats_imu, quats_body):
        q_i = normalize_quat(qi)
        q_b = normalize_quat(qb)
        if q_i is None or q_b is None:
            continue
        r_i = R.from_quat(q_i)
        r_b = R.from_quat(q_b)
        rel.append(r_i.inv() * r_b)
    if len(rel) < 3:
        raise ValueError("Too few valid quaternion pairs for extrinsic estimate")
    r_mean = R.concatenate(rel).mean()
    residuals = []
    for r_ib in rel:
        r_err = r_ib * r_mean.inv()
        residuals.append(float(r_err.magnitude() * 180.0 / math.pi))
    rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
    return ImuExtrinsic(
        quat_imu_to_body=r_mean.as_quat(),
        n_samples=len(rel),
        residual_angle_deg_rms=rms,
    )


def estimate_from_timestamped_quats(
    t_imu: np.ndarray,
    quats_imu: Sequence[np.ndarray],
    t_body: np.ndarray,
    quats_body: Sequence[np.ndarray],
    *,
    max_time_offset_sec: float = 0.05,
) -> ImuExtrinsic:
    t_imu = np.asarray(t_imu, dtype=float)
    t_body = np.asarray(t_body, dtype=float)
    if t_imu.size == 0 or t_body.size == 0:
        raise ValueError("Empty timestamp arrays")
    pick = nearest_indices(t_imu, t_body)
    dt = np.abs(t_body[pick] - t_imu)
    mask = dt <= max_time_offset_sec
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError(
            f"Only {int(np.count_nonzero(mask))} pairs within {max_time_offset_sec}s sync; need >= 3"
        )
    qi = [quats_imu[i] for i in range(len(quats_imu)) if mask[i]]
    qb = [quats_body[pick[i]] for i in range(len(quats_imu)) if mask[i]]
    return estimate_imu_extrinsic(qi, qb)


def resolve_extrinsic_path(
    path: Optional[Union[str, Path]],
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    if path is None or str(path).strip() == "":
        return default_extrinsic_path(base_dir)
    p = Path(path).expanduser()
    if not p.is_absolute():
        root = base_dir if base_dir is not None else Path(__file__).resolve().parent
        p = root / p
    return p


def load_extrinsic_for_node(
    path: Optional[Union[str, Path]],
    *,
    enabled: bool,
    required: bool = False,
    node_name: str = "node",
) -> Optional[ImuExtrinsic]:
    if not enabled:
        return None
    resolved = resolve_extrinsic_path(path)
    ext = load_imu_extrinsic(resolved, required=required)
    if ext is None:
        import rospy

        rospy.logwarn(
            "%s: imu extrinsic enabled but file missing (%s); using raw IMU.",
            node_name,
            str(resolved),
        )
    return ext
