#!/usr/bin/env python3
"""
Calibrate CDPR kinematic parameters from mocap, IMU Euler angles and encoders.

Data file format, one sample per line:
    x y z yaw pitch roll theta1 theta2 ... theta8

The script estimates 57 parameters:
    l0      : 8 cable original lengths
    a       : 8 frame anchor points in world/base coordinates, shape (8, 3)
    b       : 8 platform anchor points in end-effector coordinates, shape (8, 3)
    yaw0    : IMU initial heading offset

Kinematic model for cable i at sample k:
    residual_i = ||a_i - Rz(yaw + yaw0) Ry(pitch) Rx(roll) b_i - p|| - l0_i - r theta_i

Euler convention:
    The input line stores Euler angles as yaw, pitch, roll.
    "ZYX intrinsic" rotation is implemented as:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    The calibrated heading offset yaw0 is added only to yaw.

Example:
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.020 --output calib.json
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.020 --ypr-degrees
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares


N_CABLES = 8
N_PARAMS = 8 + 8 * 3 + 8 * 3 + 1


# Fallback nominal geometry from cdpr.py / cdpr_euler_ekf.py.
# It is only an initial guess; LM will update these values.
DEFAULT_A = np.array(
    [
        [-0.260, -0.243, 2.300],
        [-0.361, -0.125, 2.300],
        [-2.049, -0.089, 2.300],
        [-2.169, -0.212, 2.300],
        [-2.193, -1.225, 2.290],
        [-2.084, -1.357, 2.300],
        [-0.415, -1.384, 2.300],
        [-0.290, -1.252, 2.300],
    ],
    dtype=float,
)

DEFAULT_B = np.array(
    [
        [0.0184, -0.0125, 0.0110],
        [-0.0140, 0.0169, -0.0110],
        [0.0140, 0.0169, 0.0110],
        [-0.0184, -0.0125, -0.0110],
        [-0.0184, 0.0125, 0.0110],
        [0.0140, -0.0169, -0.0110],
        [-0.0140, -0.0169, 0.0110],
        [0.0184, 0.0125, -0.0110],
    ],
    dtype=float,
)


def rot_x(angle: float) -> np.ndarray:
    """Rotation about x axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation about y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation about z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def drot_z(angle: float) -> np.ndarray:
    """Derivative d(Rz(angle))/d(angle)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])


def ypr_zyx_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return R from end-effector frame to world frame for intrinsic ZYX Euler angles."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def dypr_zyx_dyaw(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Derivative of Rz(yaw) @ Ry(pitch) @ Rx(roll) with respect to yaw."""
    return drot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_samples(path: Path, ypr_degrees: bool, theta_degrees: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load calibration samples.

    Returns:
        p:     (k, 3) mocap positions
        ypr:   (k, 3) IMU yaw, pitch, roll
        theta: (k, 8) encoder angles
    """
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    expected_cols = 3 + 3 + N_CABLES
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"{path} should have {expected_cols} columns: "
            "x y z yaw pitch roll theta1 ... theta8"
        )

    p = data[:, 0:3].astype(float)
    ypr = data[:, 3:6].astype(float)
    theta = data[:, 6:14].astype(float)
    if ypr_degrees:
        ypr = np.deg2rad(ypr)
    if theta_degrees:
        theta = np.deg2rad(theta)
    return p, ypr, theta


def unpack_params(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert the flat 57-vector into l0, a, b and yaw0."""
    l0 = x[0:8]
    a = x[8:32].reshape(N_CABLES, 3)
    b = x[32:56].reshape(N_CABLES, 3)
    yaw0 = float(x[56])
    return l0, a, b, yaw0


def pack_params(l0: np.ndarray, a: np.ndarray, b: np.ndarray, yaw0: float) -> np.ndarray:
    """Convert l0, a, b and yaw0 into the flat parameter vector used by LM."""
    return np.hstack([l0.reshape(-1), a.reshape(-1), b.reshape(-1), np.array([yaw0])])


def model_lengths(p: np.ndarray, ypr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute geometric cable lengths ||a - R b - p|| for all samples and cables."""
    _, a, b, yaw0 = unpack_params(x)
    lengths = np.zeros((p.shape[0], N_CABLES), dtype=float)
    for k in range(p.shape[0]):
        yaw_meas, pitch, roll = ypr[k]
        R = ypr_zyx_to_matrix(yaw_meas + yaw0, pitch, roll)
        for i in range(N_CABLES):
            d = a[i] - R @ b[i] - p[k]
            lengths[k, i] = np.linalg.norm(d)
    return lengths


def residuals(x: np.ndarray, p: np.ndarray, ypr: np.ndarray, theta: np.ndarray, radius: float) -> np.ndarray:
    """
    Residual vector stacked sample-by-sample.

    For each row and each cable:
        f = geometric_length - l0 - radius * theta
    """
    l0, _, _, _ = unpack_params(x)
    geom_lengths = model_lengths(p, ypr, x)
    return (geom_lengths - l0.reshape(1, N_CABLES) - radius * theta).reshape(-1)


def residual_jacobian(
    x: np.ndarray,
    p: np.ndarray,
    ypr: np.ndarray,
    theta: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Analytic Jacobian of residuals with respect to the 57 parameters.

    theta and radius are not differentiated because encoder readings and drum
    radius are treated as known measurements/configuration in this calibration.
    """
    del theta, radius
    _, a, b, yaw0 = unpack_params(x)
    n_samples = p.shape[0]
    J = np.zeros((n_samples * N_CABLES, N_PARAMS), dtype=float)

    for k in range(n_samples):
        yaw_meas, pitch, roll = ypr[k]
        yaw = yaw_meas + yaw0
        R = ypr_zyx_to_matrix(yaw, pitch, roll)
        dR_dyaw0 = dypr_zyx_dyaw(yaw, pitch, roll)

        for i in range(N_CABLES):
            row = k * N_CABLES + i
            d = a[i] - R @ b[i] - p[k]
            length = float(np.linalg.norm(d))
            if length < 1e-12:
                # This should never happen for a real CDPR; keep the Jacobian finite.
                unit = np.zeros(3, dtype=float)
            else:
                unit = d / length

            # l0_i enters as "-l0_i".
            J[row, i] = -1.0

            # a_i is directly inside d.
            a_col = 8 + 3 * i
            J[row, a_col:a_col + 3] = unit

            # b_i enters through -R b_i.
            b_col = 32 + 3 * i
            J[row, b_col:b_col + 3] = -(unit @ R)

            # yaw0 only changes yaw, so d/dyaw0 ||d|| = unit^T (-dR/dyaw b_i).
            J[row, 56] = -float(unit @ (dR_dyaw0 @ b[i]))

    return J


def params_to_dict(x: np.ndarray, cost_info: Dict[str, object] = None) -> Dict[str, object]:
    """Make a JSON-friendly result dictionary."""
    l0, a, b, yaw0 = unpack_params(x)
    out = {
        "l0": l0.tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "yaw0": wrap_angle(yaw0),
        "yaw0_raw": yaw0,
    }
    if cost_info:
        out.update(cost_info)
    return out


def load_initial_json(path: Path) -> Dict[str, object]:
    """Load optional initial values from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_initial_guess(
    p: np.ndarray,
    ypr: np.ndarray,
    theta: np.ndarray,
    radius: float,
    init_path: Path = None,
) -> np.ndarray:
    """
    Build the LM initial value.

    If --init is not provided, use the nominal CDPR geometry and estimate l0 by
    averaging geometric_length - radius * theta over all samples. This makes the
    initial residual mean close to zero for every cable.
    """
    a = DEFAULT_A.copy()
    b = DEFAULT_B.copy()
    yaw0 = 0.0
    l0 = None

    if init_path is not None:
        init = load_initial_json(init_path)
        if "a" in init:
            a = np.asarray(init["a"], dtype=float).reshape(N_CABLES, 3)
        if "b" in init:
            b = np.asarray(init["b"], dtype=float).reshape(N_CABLES, 3)
        if "yaw0" in init:
            yaw0 = float(init["yaw0"])
        if "l0" in init:
            l0 = np.asarray(init["l0"], dtype=float).reshape(N_CABLES)

    if l0 is None:
        x_tmp = pack_params(np.zeros(N_CABLES, dtype=float), a, b, yaw0)
        geom_lengths = model_lengths(p, ypr, x_tmp)
        l0 = np.mean(geom_lengths - radius * theta, axis=0)

    return pack_params(l0, a, b, yaw0)


def summarize_residuals(res: np.ndarray) -> Dict[str, object]:
    """Compute residual statistics in meters."""
    res_by_cable = res.reshape(-1, N_CABLES)
    rmse_all = float(np.sqrt(np.mean(res**2)))
    rmse_by_cable = np.sqrt(np.mean(res_by_cable**2, axis=0))
    return {
        "rmse_m": rmse_all,
        "max_abs_m": float(np.max(np.abs(res))),
        "rmse_by_cable_m": rmse_by_cable.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate 8-cable CDPR kinematic parameters with LM least squares."
    )
    parser.add_argument("data", type=Path, help="Text file: x y z yaw pitch roll theta1 ... theta8 per line.")
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Drum radius r in meters. Encoder theta is assumed to be radians unless --theta-degrees is set.",
    )
    parser.add_argument("--init", type=Path, default=None, help="Optional JSON initial guess.")
    parser.add_argument("--output", type=Path, default=Path("cdpr_kinematic_calib.json"), help="Output JSON path.")
    parser.add_argument("--ypr-degrees", action="store_true", help="Treat input yaw/pitch/roll as degrees.")
    parser.add_argument("--theta-degrees", action="store_true", help="Treat input encoder theta as degrees.")
    parser.add_argument("--max-nfev", type=int, default=2000, help="Maximum LM function evaluations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p, ypr, theta = load_samples(args.data, args.ypr_degrees, args.theta_degrees)

    n_residuals = p.shape[0] * N_CABLES
    if n_residuals < N_PARAMS:
        raise ValueError(
            f"Need at least {N_PARAMS} residuals for LM, got {n_residuals}. "
            "Use at least 8 calibration poses, and preferably many more with diverse motion."
        )

    x0 = make_initial_guess(p, ypr, theta, args.radius, args.init)
    res0 = residuals(x0, p, ypr, theta, args.radius)
    print(f"Loaded {p.shape[0]} samples, {n_residuals} residuals, {N_PARAMS} parameters.")
    print(f"Initial RMSE: {np.sqrt(np.mean(res0 ** 2)):.6f} m")

    # method="lm" selects Levenberg-Marquardt. It does not support bounds, so
    # good initial values and rich excitation in the calibration data matter.
    result = least_squares(
        residuals,
        x0,
        jac=residual_jacobian,
        method="lm",
        args=(p, ypr, theta, args.radius),
        max_nfev=args.max_nfev,
        x_scale="jac",
    )

    res_final = residuals(result.x, p, ypr, theta, args.radius)
    stats = summarize_residuals(res_final)
    result_dict = params_to_dict(
        result.x,
        {
            "success": bool(result.success),
            "message": result.message,
            "n_samples": int(p.shape[0]),
            "n_residuals": int(n_residuals),
            "n_parameters": int(N_PARAMS),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "initial_rmse_m": float(np.sqrt(np.mean(res0**2))),
            **stats,
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
        f.write("\n")

    print(f"LM success: {result.success} ({result.message})")
    print(f"Final RMSE: {stats['rmse_m']:.6f} m")
    print(f"Max |residual|: {stats['max_abs_m']:.6f} m")
    print(f"yaw0: {result_dict['yaw0']:.9f} rad ({math.degrees(result_dict['yaw0']):.6f} deg)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
