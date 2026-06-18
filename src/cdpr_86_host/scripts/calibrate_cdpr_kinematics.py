#!/usr/bin/env python3
"""
Calibrate CDPR kinematic parameters from mocap, IMU Euler angles and encoders.

Data file format, one sample per line:
    x y z yaw pitch roll theta1 theta2 ... theta8

Optional metadata line, usually written by record_cdpr_calib_data.py:
    # cdpr_calib_metadata {"init_motor_pos_abs":[...]}

Use --rows to calibrate on a subset of loaded numeric rows (0-based slice list; see --rows help).
Use --calibrate-params to choose whether a, b, yaw0 and r are optimized.

The script estimates 65 parameters:
    l0      : 8 cable original lengths
    a       : 8 frame anchor points in world/base coordinates, shape (8, 3)
    b       : 8 platform anchor points in end-effector coordinates, shape (8, 3)
    yaw0    : IMU initial heading offset
    r       : 8 drum radii, one per motor/cable

Kinematic model for cable i at sample k:
    residual_i = ||a_i - Rz(yaw + yaw0) Ry(pitch) Rx(roll) b_i - p|| - l0_i - s_i r_i theta_i

Motor sign convention follows cdpr.py:
    s = [-1, +1, -1, +1, -1, +1, -1, +1]

Euler convention:
    The input line stores Euler angles as yaw, pitch, roll.
    "ZYX intrinsic" rotation is implemented as:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    The calibrated heading offset yaw0 is added only to yaw.

Example:
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.020 --output calib.json
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.020 --ypr-degrees
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.025 --rows "1:3,10:20,-5:"
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.025 --rows "[1:3，5：13，-13：-1]"
    python3 calibrate_cdpr_kinematics.py samples.txt --radius 0.025 --calibrate-params '{"a": false, "b": true, "yaw0": false, "r": true}'
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares


N_CABLES = 8
N_PARAMS = 8 + 8 * 3 + 8 * 3 + 1 + 8
MOTOR_TO_LENGTH_SIGN = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=float)
DEFAULT_CALIBRATE_PARAMS = {"a": True, "b": True, "yaw0": True, "r": True}


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
        [0.184, -0.125, 0.110],
        [-0.140, 0.169, -0.110],
        [0.140, 0.169, 0.110],
        [-0.184, -0.125, -0.110],
        [-0.184, 0.125, 0.110],
        [0.140, -0.169, -0.110],
        [-0.140, -0.169, 0.110],
        [0.184, 0.125, -0.110],
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


def parse_row_indices(spec: str, n_rows: int) -> np.ndarray:
    """
    Build a sorted list of unique row indices from a comma-separated slice spec.

    Indices follow Python/NumPy rules on the loaded numeric sample matrix (0-based,
    half-open [start:end), negative indices count from the end).

    The string may use ASCII or full-width punctuation; optional surrounding [...]
    and spaces are stripped. Examples (n_rows large enough):
        "1:3"       -> rows 1, 2
        "5:13"      -> rows 5 .. 12
        "-13:-1"    -> rows n-13 .. n-2
        "[1:3,10:20,-5:]" -> same with brackets
        "10：20"    -> rows 10..19 (full-width colon)
    """
    spec = spec.strip()
    if spec.startswith("[") and spec.endswith("]"):
        spec = spec[1:-1].strip()
    # Normalize full-width punctuation (common when copying from docs)
    spec = spec.replace("，", ",").replace("：", ":").replace(" ", "")
    if not spec:
        return np.arange(n_rows, dtype=int)

    ar = np.arange(n_rows, dtype=int)
    chunks: List[np.ndarray] = []

    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if part.count(":") > 1:
            raise ValueError(f"Invalid row segment (at most one ':'): {raw!r}")
        if ":" in part:
            left, right = part.split(":", 1)
            start = int(left) if left != "" else None
            end = int(right) if right != "" else None
            idx = ar[slice(start, end)]
        else:
            k = int(part)
            if k < 0:
                k += n_rows
            if k < 0 or k >= n_rows:
                raise ValueError(f"Row index {part!r} out of range for n_rows={n_rows}")
            idx = np.array([k], dtype=int)
        if idx.size == 0:
            raise ValueError(f"Row segment {raw!r} selects no rows (n_rows={n_rows}).")
        chunks.append(idx)

    if not chunks:
        return np.arange(n_rows, dtype=int)

    out = np.unique(np.concatenate(chunks))
    if out.size == 0:
        raise ValueError("Row selection is empty.")
    return out


def parse_calibrate_params(spec: str) -> Dict[str, bool]:
    """Parse a JSON dict selecting which parameter groups to optimize."""
    if not spec:
        return DEFAULT_CALIBRATE_PARAMS.copy()
    try:
        raw = json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--calibrate-params should be a JSON dict: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("--calibrate-params should be a JSON dict with keys: a, b, yaw0, r")

    allowed = set(DEFAULT_CALIBRATE_PARAMS)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown --calibrate-params keys: {unknown}; allowed keys: {sorted(allowed)}")

    out = DEFAULT_CALIBRATE_PARAMS.copy()
    for key, value in raw.items():
        if not isinstance(value, bool):
            raise ValueError(f"--calibrate-params[{key!r}] should be true or false, got {value!r}")
        out[key] = value
    return out


def load_sample_metadata(path: Path) -> Dict[str, object]:
    """Read optional JSON metadata from a comment line in the calibration txt."""
    prefix = "# cdpr_calib_metadata "
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            if stripped.startswith(prefix):
                return json.loads(stripped[len(prefix):])
    return {}


def normalized_init_motor_pos_abs(metadata: Dict[str, object]) -> object:
    """Return init_motor_pos_abs as a JSON-safe list, or None when unavailable."""
    if "init_motor_pos_abs" not in metadata:
        return None
    init_motor_pos_abs = np.asarray(metadata["init_motor_pos_abs"], dtype=float).reshape(-1)
    if init_motor_pos_abs.size != N_CABLES:
        raise ValueError(f"init_motor_pos_abs should have {N_CABLES} values, got {init_motor_pos_abs.size}")
    return init_motor_pos_abs.tolist()


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


def unpack_params(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Convert the flat parameter vector into l0, a, b, yaw0 and r."""
    l0 = x[0:8]
    a = x[8:32].reshape(N_CABLES, 3)
    b = x[32:56].reshape(N_CABLES, 3)
    yaw0 = float(x[56])
    r = x[57:65]
    return l0, a, b, yaw0, r


def pack_params(l0: np.ndarray, a: np.ndarray, b: np.ndarray, yaw0: float, r: np.ndarray) -> np.ndarray:
    """Convert l0, a, b, yaw0 and r into the flat parameter vector used by LM."""
    return np.hstack([l0.reshape(-1), a.reshape(-1), b.reshape(-1), np.array([yaw0]), r.reshape(-1)])


def active_parameter_indices(calibrate_params: Dict[str, bool]) -> np.ndarray:
    """Return flat parameter indices optimized by LM. l0 is always calibrated."""
    chunks = [np.arange(0, 8, dtype=int)]
    if calibrate_params["a"]:
        chunks.append(np.arange(8, 32, dtype=int))
    if calibrate_params["b"]:
        chunks.append(np.arange(32, 56, dtype=int))
    if calibrate_params["yaw0"]:
        chunks.append(np.array([56], dtype=int))
    if calibrate_params["r"]:
        chunks.append(np.arange(57, 65, dtype=int))
    return np.concatenate(chunks)


def expand_active_params(x_active: np.ndarray, x_reference: np.ndarray, active_idx: np.ndarray) -> np.ndarray:
    """Fill active values into a full parameter vector, keeping inactive groups fixed."""
    x_full = x_reference.copy()
    x_full[active_idx] = x_active
    return x_full


def model_lengths(p: np.ndarray, ypr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute geometric cable lengths ||a - R b - p|| for all samples and cables."""
    _, a, b, yaw0, _ = unpack_params(x)
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
        f = geometric_length - l0 - motor_to_length_sign * r_i * theta
    """
    del radius
    l0, _, _, _, r = unpack_params(x)
    geom_lengths = model_lengths(p, ypr, x)
    encoder_lengths = theta * (MOTOR_TO_LENGTH_SIGN * r).reshape(1, N_CABLES)
    return (geom_lengths - l0.reshape(1, N_CABLES) - encoder_lengths).reshape(-1)


def residual_jacobian(
    x: np.ndarray,
    p: np.ndarray,
    ypr: np.ndarray,
    theta: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Analytic Jacobian of residuals with respect to the full parameter vector.

    Encoder readings are measurements; drum radii are parameters in x.
    """
    del radius
    _, a, b, yaw0, _ = unpack_params(x)
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

            # r_i enters as "-s_i * r_i * theta_i".
            J[row, 57 + i] = -MOTOR_TO_LENGTH_SIGN[i] * theta[k, i]

    return J


def residuals_active(
    x_active: np.ndarray,
    x_reference: np.ndarray,
    active_idx: np.ndarray,
    p: np.ndarray,
    ypr: np.ndarray,
    theta: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Residuals for the active optimization vector."""
    x_full = expand_active_params(x_active, x_reference, active_idx)
    return residuals(x_full, p, ypr, theta, radius)


def residual_jacobian_active(
    x_active: np.ndarray,
    x_reference: np.ndarray,
    active_idx: np.ndarray,
    p: np.ndarray,
    ypr: np.ndarray,
    theta: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Jacobian projected to the active optimization vector."""
    x_full = expand_active_params(x_active, x_reference, active_idx)
    return residual_jacobian(x_full, p, ypr, theta, radius)[:, active_idx]


def params_to_dict(x: np.ndarray, cost_info: Dict[str, object] = None) -> Dict[str, object]:
    """Make a JSON-friendly result dictionary."""
    l0, a, b, yaw0, r = unpack_params(x)
    out = {
        "l0": l0.tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "yaw0": wrap_angle(yaw0),
        "yaw0_raw": yaw0,
        "r": r.tolist(),
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
    calibrate_params: Dict[str, bool] = None,
) -> np.ndarray:
    """
    Build the LM initial value.

    If --init is not provided, use the nominal CDPR geometry and --radius as
    r1..r8. Estimate l0 by averaging geometric_length - motor_to_length_sign *
    r_i * theta over all samples. This makes the initial residual mean close to
    zero for every cable.
    """
    a = DEFAULT_A.copy()
    b = DEFAULT_B.copy()
    yaw0 = 0.0
    r = np.full(N_CABLES, float(radius), dtype=float)
    l0 = None
    calibrate_params = calibrate_params or DEFAULT_CALIBRATE_PARAMS

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
        if "r" in init:
            r = np.asarray(init["r"], dtype=float).reshape(N_CABLES)
        elif "radius" in init:
            radius_init = np.asarray(init["radius"], dtype=float).reshape(-1)
            if radius_init.size == 1:
                r = np.full(N_CABLES, float(radius_init[0]), dtype=float)
            elif radius_init.size == N_CABLES:
                r = radius_init.astype(float)
            else:
                raise ValueError(f"init radius should have 1 or {N_CABLES} values, got {radius_init.size}")

    if not calibrate_params["yaw0"]:
        yaw0 = 0.0

    if l0 is None:
        x_tmp = pack_params(np.zeros(N_CABLES, dtype=float), a, b, yaw0, r)
        geom_lengths = model_lengths(p, ypr, x_tmp)
        encoder_lengths = theta * (MOTOR_TO_LENGTH_SIGN * r).reshape(1, N_CABLES)
        l0 = np.mean(geom_lengths - encoder_lengths, axis=0)

    return pack_params(l0, a, b, yaw0, r)


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
        help=(
            "Reference drum radius in meters. Used as r1..r8 initial/reference values; "
            "encoder theta is assumed radians unless --theta-degrees is set."
        ),
    )
    parser.add_argument("--init", type=Path, default=None, help="Optional JSON initial guess.")
    parser.add_argument("--output", type=Path, default=Path("cdpr_kinematic_calib.json"), help="Output JSON path.")
    parser.add_argument("--ypr-degrees", action="store_true", help="Treat input yaw/pitch/roll as degrees.")
    parser.add_argument("--theta-degrees", action="store_true", help="Treat input encoder theta as degrees.")
    parser.add_argument("--max-nfev", type=int, default=2000, help="Maximum LM function evaluations.")
    parser.add_argument(
        "--rows",
        type=str,
        default="",
        help=(
            "Comma-separated slice ranges over loaded sample rows (0-based, half-open [start:end), "
            "Python/NumPy rules; negative indices from end). Optional [...] wrapper; ASCII or full-width ，： "
            "Example: '1:3,10:20,-5:' or '[1:3，10：20，-5：]' Default: all rows."
        ),
    )
    parser.add_argument(
        "--calibrate-params",
        type=str,
        default=json.dumps(DEFAULT_CALIBRATE_PARAMS),
        help=(
            "JSON dict selecting optimized groups among a, b, yaw0. Missing keys default true. "
            "l0 is always calibrated. If a/b is false, it stays at the reference initial value; "
            "if yaw0 is false, it stays at 0; if r is false, it stays at --radius or init r. "
            "Example: '{\"a\": false, \"b\": true, \"yaw0\": false, \"r\": true}'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrate_params = parse_calibrate_params(args.calibrate_params)
    metadata = load_sample_metadata(args.data)
    init_motor_pos_abs = normalized_init_motor_pos_abs(metadata)
    p, ypr, theta = load_samples(args.data, args.ypr_degrees, args.theta_degrees)
    n_rows_loaded = int(p.shape[0])

    row_spec = (args.rows or "").strip()
    if row_spec:
        row_idx = parse_row_indices(row_spec, n_rows_loaded)
        p = p[row_idx]
        ypr = ypr[row_idx]
        theta = theta[row_idx]
        if row_idx.size <= 40:
            idx_str = str(row_idx.tolist())
        else:
            idx_str = f"{row_idx[:20].tolist()} ... {row_idx[-10:].tolist()} ({row_idx.size} total)"
        print(f"Row selection {row_spec!r}: using {p.shape[0]} / {n_rows_loaded} samples (indices: {idx_str})")
    else:
        row_idx = np.arange(n_rows_loaded, dtype=int)
        print(f"Using all {n_rows_loaded} samples.")

    n_residuals = p.shape[0] * N_CABLES
    active_idx = active_parameter_indices(calibrate_params)
    n_active_params = int(active_idx.size)
    if n_residuals < n_active_params:
        raise ValueError(
            f"Need at least {n_active_params} residuals for LM, got {n_residuals}. "
            "Use at least 8 calibration poses, and preferably many more with diverse motion."
        )

    x0 = make_initial_guess(p, ypr, theta, args.radius, args.init, calibrate_params)
    x0_active = x0[active_idx]
    res0 = residuals(x0, p, ypr, theta, args.radius)
    print(
        f"Loaded {p.shape[0]} samples, {n_residuals} residuals, "
        f"{n_active_params} active parameters ({N_PARAMS} full parameters)."
    )
    print(f"Calibrate params: {calibrate_params}")
    print(f"Initial RMSE: {np.sqrt(np.mean(res0 ** 2)):.6f} m")

    # method="lm" selects Levenberg-Marquardt. It does not support bounds, so
    # good initial values and rich excitation in the calibration data matter.
    result = least_squares(
        residuals_active,
        x0_active,
        jac=residual_jacobian_active,
        method="lm",
        args=(x0, active_idx, p, ypr, theta, args.radius),
        max_nfev=args.max_nfev,
        x_scale="jac",
    )

    x_final = expand_active_params(result.x, x0, active_idx)
    res_final = residuals(x_final, p, ypr, theta, args.radius)
    stats = summarize_residuals(res_final)
    result_dict = params_to_dict(
        x_final,
        {
            "success": bool(result.success),
            "message": result.message,
            "calibrate_params": calibrate_params,
            "init_motor_pos_abs": init_motor_pos_abs,
            "motor_to_length_sign": MOTOR_TO_LENGTH_SIGN.tolist(),
            "n_rows_loaded": n_rows_loaded,
            "row_selection": row_spec if row_spec else "all",
            "row_indices": row_idx.tolist(),
            "n_samples": int(p.shape[0]),
            "n_residuals": int(n_residuals),
            "n_parameters": int(n_active_params),
            "n_parameters_full": int(N_PARAMS),
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
    print(f"r: {result_dict['r']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
