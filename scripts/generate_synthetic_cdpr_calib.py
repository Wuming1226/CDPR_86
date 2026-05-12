#!/usr/bin/env python3
"""
Generate synthetic CDPR calibration records for testing calibrate_cdpr_kinematics.py.

Uses DEFAULT_A / DEFAULT_B and MOTOR_TO_LENGTH_SIGN from calibrate_cdpr_kinematics.py.
For each sample:
  - True pose: position p_true and intrinsic ZYX Euler (yaw_true, pitch, roll).
  - Geometric cable length: L_i = ||a_i - R(yaw_true, pitch, roll) b_i - p_true||.
  - l0 is fixed from the first sample's true lengths (so sample 0 has theta = 0 before noise).
  - Noiseless encoder: theta_i = (L_i - l0_i) / (s_i * r).
  - IMU reports yaw with a fixed bias: yaw_recorded = wrap(yaw_true - yaw_offset_rad)
    so that calibration should recover yaw0 ≈ +yaw_offset (default +10 deg).

Output: one line per sample (default 80), same as record_cdpr_calib_data.txt:
    x y z yaw pitch roll theta1 ... theta8
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from calibrate_cdpr_kinematics import (  # noqa: E402
    DEFAULT_A,
    DEFAULT_B,
    MOTOR_TO_LENGTH_SIGN,
    N_CABLES,
    ypr_zyx_to_matrix,
)

# Synthetic pose law (reference / standard values, printed each run)
WORKSPACE_CENTER_OFFSET = np.array([0.0, 0.0, -0.35], dtype=float)
POS_LISSAJOUS_AX = 0.22
POS_LISSAJOUS_AY = 0.18
POS_LISSAJOUS_AZ = 0.12
POS_RAND_SIGMA = np.array([0.05, 0.05, 0.04], dtype=float)
PITCH_ROLL_LOW = -0.12
PITCH_ROLL_HIGH = 0.12
LISSAJOUS_FREQ_X = 2.0
LISSAJOUS_FREQ_Y = 1.7
LISSAJOUS_FREQ_Z = 3.1


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def geom_lengths(
    p: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Cable lengths (k, 8) for batches of ZYX Euler poses."""
    k = p.shape[0]
    out = np.zeros((k, N_CABLES), dtype=float)
    for j in range(k):
        R = ypr_zyx_to_matrix(float(yaw[j]), float(pitch[j]), float(roll[j]))
        for i in range(N_CABLES):
            d = a[i] - R @ b[i] - p[j]
            out[j, i] = float(np.linalg.norm(d))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic CDPR calibration txt (80 rows by default).")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("cabli_record.txt"),
        help="Output text path (14 columns per line).",
    )
    p.add_argument("--n-samples", type=int, default=80, help="Number of rows (default 80).")
    p.add_argument("--radius", type=float, default=0.025, help="Drum radius r [m], same as cdpr.py.")
    p.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=10.0,
        help="True IMU heading bias: recorded_yaw = yaw_true - this (deg); calibrate should find yaw0 ≈ +this.",
    )
    p.add_argument("--pos-noise-m", type=float, default=0.002, help="Std of Gaussian noise on recorded x,y,z [m].")
    p.add_argument("--ypr-noise-deg", type=float, default=0.3, help="Std of Gaussian noise on pitch, roll [deg].")
    p.add_argument("--theta-noise-deg", type=float, default=0.05, help="Std of Gaussian noise on each theta [deg].")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return p.parse_args()


def print_standard_parameter_report(
    args: argparse.Namespace,
    a_mean: np.ndarray,
    center: np.ndarray,
    l0: np.ndarray,
) -> None:
    """Print CLI values, geometry from calibrate_cdpr_kinematics, pose law constants, and sample-0 l0."""
    r = float(args.radius)
    yaw_off_rad = math.radians(float(args.yaw_offset_deg))

    print("=== synthetic CDPR calib: standard / effective parameters ===")
    print("[CLI]")
    print(f"  output              = {args.output}")
    print(f"  n_samples           = {args.n_samples}")
    print(f"  radius r [m]        = {r}")
    print(f"  yaw_offset_deg      = {args.yaw_offset_deg}  (recorded_yaw = wrap(yaw_true - this); expect yaw0 ≈ +this in calibrate)")
    print(f"  yaw_offset_rad      = {yaw_off_rad}")
    print(f"  pos_noise_m (std)   = {args.pos_noise_m}")
    print(f"  ypr_noise_deg (std) = {args.ypr_noise_deg}  (pitch & roll only)")
    print(f"  theta_noise_deg     = {args.theta_noise_deg}")
    print(f"  seed                = {args.seed}")

    print("[geometry: DEFAULT_A (8×3), world / base frame]")
    print(np.array2string(DEFAULT_A, precision=6, suppress_small=False))

    print("[geometry: DEFAULT_B (8×3), body frame]")
    print(np.array2string(DEFAULT_B, precision=6, suppress_small=False))

    print("[geometry: MOTOR_TO_LENGTH_SIGN s_i, matches cdpr.py / calibrate]")
    print(f"  {MOTOR_TO_LENGTH_SIGN.tolist()}")

    print("[pose generator: workspace & Lissajous / RNG]")
    print(f"  a_mean (from DEFAULT_A)     = {a_mean.tolist()}")
    print(f"  WORKSPACE_CENTER_OFFSET     = {WORKSPACE_CENTER_OFFSET.tolist()}")
    print(f"  center = a_mean + offset    = {center.tolist()}")
    print(f"  POS_LISSAJOUS_AX/Y/Z        = {POS_LISSAJOUS_AX}, {POS_LISSAJOUS_AY}, {POS_LISSAJOUS_AZ}")
    print(f"  LISSAJOUS_FREQ_X/Y/Z (×π)   = {LISSAJOUS_FREQ_X}, {LISSAJOUS_FREQ_Y}, {LISSAJOUS_FREQ_Z}")
    print(f"  POS_RAND_SIGMA (x,y,z)      = {POS_RAND_SIGMA.tolist()}")
    print(f"  pitch, roll uniform [rad] = [{PITCH_ROLL_LOW}, {PITCH_ROLL_HIGH}]")
    print(f"  yaw_true uniform [rad]      = [-pi, pi]")

    print("[reference cable lengths l0: geometric lengths at sample 0, theta=0 before noise]")
    print(f"  l0 [m] = {l0.tolist()}")
    print("=== end parameter report ===")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    a = DEFAULT_A.copy()
    b = DEFAULT_B.copy()
    r = float(args.radius)
    yaw_off = math.radians(float(args.yaw_offset_deg))
    n = int(args.n_samples)

    # Rough workspace center from anchor geometry (feasible hanging pose band).
    a_mean = a.mean(axis=0)
    center = a_mean + WORKSPACE_CENTER_OFFSET

    p_true = np.zeros((n, 3), dtype=float)
    yaw_true = np.zeros(n, dtype=float)
    pitch_true = np.zeros(n, dtype=float)
    roll_true = np.zeros(n, dtype=float)

    for j in range(n):
        t = j / max(n - 1, 1)
        p_true[j] = center + np.array(
            [
                POS_LISSAJOUS_AX * math.sin(LISSAJOUS_FREQ_X * math.pi * t)
                + POS_RAND_SIGMA[0] * rng.standard_normal(),
                POS_LISSAJOUS_AY * math.cos(LISSAJOUS_FREQ_Y * math.pi * t)
                + POS_RAND_SIGMA[1] * rng.standard_normal(),
                POS_LISSAJOUS_AZ * math.sin(LISSAJOUS_FREQ_Z * math.pi * t)
                + POS_RAND_SIGMA[2] * rng.standard_normal(),
            ],
            dtype=float,
        )
        yaw_true[j] = rng.uniform(-math.pi, math.pi)
        pitch_true[j] = rng.uniform(PITCH_ROLL_LOW, PITCH_ROLL_HIGH)
        roll_true[j] = rng.uniform(PITCH_ROLL_LOW, PITCH_ROLL_HIGH)

    L = geom_lengths(p_true, yaw_true, pitch_true, roll_true, a, b)
    if np.any(L < 1e-6):
        raise RuntimeError("A cable length collapsed; loosen pose box or check geometry.")

    l0 = L[0].copy()
    print_standard_parameter_report(args, a_mean, center, l0)

    sign = MOTOR_TO_LENGTH_SIGN.reshape(1, N_CABLES)
    theta_clean = (L - l0.reshape(1, N_CABLES)) / (sign * r)

    pos_noise = rng.normal(0.0, args.pos_noise_m, size=(n, 3))
    ypr_noise_rad = np.deg2rad(args.ypr_noise_deg)
    pitch_noise = rng.normal(0.0, ypr_noise_rad, size=n)
    roll_noise = rng.normal(0.0, ypr_noise_rad, size=n)
    theta_noise = np.deg2rad(rng.normal(0.0, args.theta_noise_deg, size=(n, N_CABLES)))

    p_rec = p_true + pos_noise
    yaw_rec = wrap_angle(yaw_true - yaw_off)
    pitch_rec = pitch_true + pitch_noise
    roll_rec = roll_true + roll_noise
    theta_rec = theta_clean + theta_noise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(
            "# Synthetic CDPR calib: x y z yaw pitch roll theta1..theta8 (rad)\n"
            f"# n_samples={n} radius={r} yaw_offset_deg={args.yaw_offset_deg} "
            f"pos_noise_m={args.pos_noise_m} ypr_noise_deg={args.ypr_noise_deg} "
            f"theta_noise_deg={args.theta_noise_deg} seed={args.seed}\n"
        )
        for j in range(n):
            row = np.hstack(
                [
                    p_rec[j],
                    np.array([yaw_rec[j], pitch_rec[j], roll_rec[j]], dtype=float),
                    theta_rec[j],
                ]
            )
            f.write(" ".join(f"{v:.10g}" for v in row))
            f.write("\n")

    print(f"Wrote {n} lines to {args.output}")
    print("Sanity: run calibrate with same radius, e.g.")
    print(f"  python3 {_SCRIPT_DIR / 'calibrate_cdpr_kinematics.py'} {args.output} --radius {r}")


if __name__ == "__main__":
    main()
