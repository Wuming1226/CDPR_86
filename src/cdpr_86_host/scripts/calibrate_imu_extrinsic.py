#!/usr/bin/env python3
"""
Estimate IMU -> platform body extrinsic from recorded quaternion pairs.

Input (from record_imu_extrinsic_data.py):
    q_imu_x q_imu_y q_imu_z q_imu_w q_mocap_x q_mocap_y q_mocap_z q_mocap_w

Output (same path as before, for EKF / CDPR nodes):
    scripts/cdpr_imu_extrinsic.json

Model: R_world_body = R_world_imu @ R_imu_to_body

Uses SO(3) least squares: minimize sum_k || Log( R_imu_k R_ib R_body_k^T ) ||^2
with scipy.optimize.least_squares on the rotation vector of R_ib.

Use --rows to calibrate on a subset of loaded sample rows (0-based slice list; same
syntax as calibrate_cdpr_kinematics.py), e.g.:

    python3 calibrate_imu_extrinsic.py --rows "0:5,7"
    python3 calibrate_imu_extrinsic.py --rows "[1:3，5：8]"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from calibrate_cdpr_kinematics import parse_row_indices
from imu_extrinsic import (
    ImuExtrinsic,
    default_extrinsic_path,
    estimate_imu_extrinsic,
    normalize_quat,
    resolve_extrinsic_path,
    save_imu_extrinsic,
)

DEFAULT_SAMPLES_FILE = "cdpr_imu_extrinsic_samples.txt"


def select_quat_pairs(
    quats_imu: List[np.ndarray],
    quats_body: List[np.ndarray],
    row_idx: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    n = len(quats_imu)
    if len(quats_body) != n:
        raise ValueError("IMU and mocap sample counts differ")
    idx = np.asarray(row_idx, dtype=int).reshape(-1)
    if idx.size < 3:
        raise ValueError(f"Need >= 3 rows after selection, got {idx.size}")
    for k in idx:
        if k < 0 or k >= n:
            raise ValueError(f"Row index {int(k)} out of range for n_rows={n}")
    return [quats_imu[int(k)] for k in idx], [quats_body[int(k)] for k in idx]


def load_quat_pairs(path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    quats_imu: List[np.ndarray] = []
    quats_body: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 floats, got {len(parts)}")
            vals = np.asarray([float(x) for x in parts], dtype=float)
            q_i = normalize_quat(vals[0:4])
            q_b = normalize_quat(vals[4:8])
            if q_i is None or q_b is None:
                raise ValueError(f"{path}:{line_no}: invalid quaternion")
            quats_imu.append(q_i)
            quats_body.append(q_b)
    if len(quats_imu) == 0:
        raise ValueError(f"No pose samples in {path}")
    return quats_imu, quats_body


def _residuals_rotvec(rotvec: np.ndarray, pairs: List[Tuple[R, R]]) -> np.ndarray:
    r_ib = R.from_rotvec(rotvec)
    errs = []
    for r_i, r_b in pairs:
        r_err = r_i * r_ib * r_b.inv()
        errs.append(r_err.as_rotvec())
    return np.concatenate(errs)


def estimate_imu_extrinsic_least_squares(
    quats_imu: List[np.ndarray],
    quats_body: List[np.ndarray],
) -> ImuExtrinsic:
    """Refine R_imu_to_body with non-linear least squares on SO(3) log residuals."""
    pairs: List[Tuple[R, R]] = []
    for qi, qb in zip(quats_imu, quats_body):
        pairs.append((R.from_quat(qi), R.from_quat(qb)))

    init = estimate_imu_extrinsic(quats_imu, quats_body)
    x0 = init.r_imu_to_body.as_rotvec()
    result = least_squares(
        _residuals_rotvec,
        x0,
        args=(pairs,),
        method="lm",
    )
    r_ib = R.from_rotvec(result.x)
    residuals_deg = []
    for r_i, r_b in pairs:
        r_err = r_i * r_ib * r_b.inv()
        residuals_deg.append(float(r_err.magnitude() * 180.0 / math.pi))
    rms = float(np.sqrt(np.mean(np.square(residuals_deg)))) if residuals_deg else 0.0
    return ImuExtrinsic(
        quat_imu_to_body=r_ib.as_quat(),
        n_samples=len(pairs),
        residual_angle_deg_rms=rms,
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Calibrate IMU extrinsic from recorded quaternion pairs.")
    p.add_argument(
        "--input",
        type=Path,
        default=script_dir / DEFAULT_SAMPLES_FILE,
        help="Recorded samples txt.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_extrinsic_path(script_dir),
        help="Output JSON (default: cdpr_imu_extrinsic.json).",
    )
    p.add_argument(
        "--rows",
        type=str,
        default="",
        help=(
            "Comma-separated slice ranges over loaded sample rows (0-based, half-open [start:end), "
            "negative indices from end). Example: '1:3,10:20,-5:' or '[1:3，5：8]'. Default: all rows."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = args.input.expanduser()
    if not in_path.is_absolute():
        in_path = Path(__file__).resolve().parent / in_path

    quats_imu_all, quats_body_all = load_quat_pairs(in_path)
    n_rows_loaded = len(quats_imu_all)

    row_spec = (args.rows or "").strip()
    if row_spec:
        row_idx = parse_row_indices(row_spec, n_rows_loaded)
        quats_imu, quats_body = select_quat_pairs(quats_imu_all, quats_body_all, row_idx)
        idx_str = np.array2string(row_idx, separator=",")
        print(f"Row selection {row_spec!r}: using {len(quats_imu)} / {n_rows_loaded} samples (indices: {idx_str})")
    else:
        row_idx = np.arange(n_rows_loaded, dtype=int)
        quats_imu, quats_body = quats_imu_all, quats_body_all
        if n_rows_loaded < 3:
            raise ValueError(f"Need >= 3 pose samples in {in_path}, got {n_rows_loaded}")
        print(f"Using all {n_rows_loaded} samples.")

    ext = estimate_imu_extrinsic_least_squares(quats_imu, quats_body)
    out_path = save_imu_extrinsic(ext, resolve_extrinsic_path(args.output))

    rpy_deg = ext.r_imu_to_body.as_euler("ZYX", degrees=True)
    print(f"Loaded {n_rows_loaded} samples from {in_path}")
    print(f"Saved extrinsic -> {out_path}")
    print(f"  n_samples={ext.n_samples}  residual_rms={ext.residual_angle_deg_rms:.4f} deg")
    print(f"  rpy_imu_to_body [deg]: roll={rpy_deg[2]:.4f} pitch={rpy_deg[1]:.4f} yaw={rpy_deg[0]:.4f}")
    print(f"  quat_imu_to_body: {ext.quat_imu_to_body.tolist()}")


if __name__ == "__main__":
    main()
