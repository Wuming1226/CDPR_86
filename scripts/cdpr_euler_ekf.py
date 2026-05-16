import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Basic SO(3) helper used by both FK Jacobian and attitude kinematics.
def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler321_to_c_ba(theta: np.ndarray) -> np.ndarray:
    # C_ba: vector in frame a -> frame b
    # 3-2-1 sequence: C_ba = C1(theta1) C2(theta2) C3(theta3)
    return rot_x(theta[0]) @ rot_y(theta[1]) @ rot_z(theta[2])


def s_theta(theta: np.ndarray) -> np.ndarray:
    # Euler-rate map: omega_ba^b = S_theta(theta) * theta_dot
    t1, t2, _ = theta
    c1, s1 = math.cos(t1), math.sin(t1)
    c2, s2 = math.cos(t2), math.sin(t2)
    return np.array(
        [
            [1.0, 0.0, -s2],
            [0.0, c1, s1 * c2],
            [0.0, -s1, c1 * c2],
        ]
    )


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_euler(theta: np.ndarray) -> np.ndarray:
    out = theta.copy()
    out[0] = wrap_angle(out[0])
    out[1] = wrap_angle(out[1])
    out[2] = wrap_angle(out[2])
    return out


@dataclass
class CDPRGeometry:
    winches_a: np.ndarray  # (m, 3) in inertial frame a
    attachments_b: np.ndarray  # (m, 3) in payload frame b

    @property
    def m(self) -> int:
        return self.winches_a.shape[0]


def cable_lengths_from_pose(
    r_zo_a: np.ndarray,
    theta_ba: np.ndarray,
    geom: CDPRGeometry,
) -> np.ndarray:
    c_ba = euler321_to_c_ba(theta_ba)
    c_ab = c_ba.T
    lengths = np.zeros(geom.m)
    for i in range(geom.m):
        d_i_a = r_zo_a + c_ab @ geom.attachments_b[i]
        diff = d_i_a - geom.winches_a[i]
        lengths[i] = np.linalg.norm(diff)
    return lengths


def fk_residual_and_jacobian(
    rho: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    # rho = [r_x, r_y, r_z, theta1, theta2, theta3]
    # fi = ||r_diwi^a|| - l_i
    # J uses analytic derivative of r_diwi^a = r + C_ab * r_diz^b - w_i.
    # Identity in the paper is for ∂(C_ba u)/∂q; here v = C_ab u with u = r_diz^b
    # constant in b. Matching ∂v/∂θ to finite differences requires
    # ∂v/∂θ = +C_ab * [u]_x * S_theta(theta) (sign opposite of the C_ba form).
    r = rho[:3]
    th = rho[3:]
    c_ba = euler321_to_c_ba(th)
    c_ab = c_ba.T
    s_th = s_theta(th)

    m = geom.m
    f = np.zeros(m)
    j = np.zeros((m, 6))
    for i in range(m):
        rdiz_b = geom.attachments_b[i]
        rdiwi_a = r + c_ab @ rdiz_b - geom.winches_a[i]
        li = np.linalg.norm(rdiwi_a)
        li_safe = max(li, 1e-12)
        u_i = rdiwi_a / li_safe
        fi = float(li - lengths[i])
        f[i] = fi

        dfi_dr = u_i  # shape (3,)
        d_ctrd_dtheta = c_ab @ skew(rdiz_b) @ s_th  # shape (3, 3)
        dfi_dtheta = (u_i.reshape(1, 3) @ d_ctrd_dtheta).reshape(3)
        j[i, :3] = dfi_dr
        j[i, 3:] = dfi_dtheta
    return f, j


def fk_xyz_residual_and_jacobian(
    r_zo_a: np.ndarray,
    theta_ba: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cable residual/Jacobian for xyz-only FK with fixed attitude."""
    c_ab = euler321_to_c_ba(theta_ba).T
    m = geom.m
    f = np.zeros(m)
    j = np.zeros((m, 3))
    for i in range(m):
        rdiwi_a = r_zo_a + c_ab @ geom.attachments_b[i] - geom.winches_a[i]
        li = np.linalg.norm(rdiwi_a)
        li_safe = max(li, 1e-12)
        u_i = rdiwi_a / li_safe
        f[i] = float(li - lengths[i])
        j[i, :] = u_i
    return f, j


def forward_kinematics_lm_xyz_with_fixed_attitude(
    r0: np.ndarray,
    theta_ba: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
    damping: float = 1e-3,
    reg_weights: np.ndarray = None,
    max_iters: int = 20,
    tol_step: float = 1e-8,
    tol_res: float = 1e-8,
) -> np.ndarray:
    """LM FK solver for position only, with attitude fixed from IMU."""
    r = r0.copy()
    if reg_weights is None:
        reg_weights = np.array([1.0, 1.0, 1.0], dtype=float)
    w = np.diag(reg_weights)
    for _ in range(max_iters):
        f, j = fk_xyz_residual_and_jacobian(r, theta_ba, lengths, geom)
        lhs = j.T @ j + damping * w
        rhs = j.T @ f
        step = np.linalg.solve(lhs, rhs)
        r_new = r - step
        if np.linalg.norm(step) < tol_step or np.linalg.norm(f) < tol_res:
            r = r_new
            break
        r = r_new
    return r


def fk_xyz_residual_and_jacobian(
    r_zo_a: np.ndarray,
    theta_ba: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cable residual/Jacobian for xyz-only FK with fixed attitude."""
    c_ab = euler321_to_c_ba(theta_ba).T
    m = geom.m
    f = np.zeros(m)
    j = np.zeros((m, 3))
    for i in range(m):
        rdiwi_a = r_zo_a + c_ab @ geom.attachments_b[i] - geom.winches_a[i]
        li = np.linalg.norm(rdiwi_a)
        li_safe = max(li, 1e-12)
        u_i = rdiwi_a / li_safe
        f[i] = float(li - lengths[i])
        j[i, :] = u_i
    return f, j


def forward_kinematics_lm_xyz_with_fixed_attitude(
    r0: np.ndarray,
    theta_ba: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
    damping: float = 1e-3,
    reg_weights: np.ndarray = None,
    max_iters: int = 20,
    tol_step: float = 1e-8,
    tol_res: float = 1e-8,
) -> np.ndarray:
    """LM FK solver for position only, with attitude fixed from IMU."""
    r = r0.copy()
    if reg_weights is None:
        reg_weights = np.array([1.0, 1.0, 1.0], dtype=float)
    w = np.diag(reg_weights)
    for _ in range(max_iters):
        f, j = fk_xyz_residual_and_jacobian(r, theta_ba, lengths, geom)
        lhs = j.T @ j + damping * w
        rhs = j.T @ f
        step = np.linalg.solve(lhs, rhs)
        r_new = r - step
        if np.linalg.norm(step) < tol_step or np.linalg.norm(f) < tol_res:
            r = r_new
            break
        r = r_new
    return r


def forward_kinematics_lm_xyz_with_fixed_attitude_and_prior(
    r0: np.ndarray,
    theta_ba: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
    r_prior: np.ndarray,
    damping: float = 1e-3,
    reg_weights: np.ndarray = None,
    prior_weights: np.ndarray = None,
    max_iters: int = 20,
    tol_step: float = 1e-8,
    tol_res: float = 1e-8,
) -> np.ndarray:
    """LM FK for position only (fixed attitude) with quadratic prior on xyz.

    Minimize: ||f(r)||^2 + ||W_p (r - r_prior)||^2
    """
    r = r0.copy()
    if reg_weights is None:
        reg_weights = np.array([1.0, 1.0, 1.0], dtype=float)
    if prior_weights is None:
        prior_weights = np.array([2.0, 2.0, 2.0], dtype=float)
    w = np.diag(reg_weights)
    wp = np.diag(prior_weights)

    for _ in range(max_iters):
        f, j = fk_xyz_residual_and_jacobian(r, theta_ba, lengths, geom)
        e_prior = r - r_prior
        lhs = j.T @ j + wp.T @ wp + damping * w
        rhs = j.T @ f + wp.T @ wp @ e_prior
        step = np.linalg.solve(lhs, rhs)
        r_new = r - step
        if np.linalg.norm(step) < tol_step or np.linalg.norm(f) < tol_res:
            r = r_new
            break
        r = r_new
    return r


def forward_kinematics_lm(
    rho0: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
    damping: float = 1e-3,
    reg_weights: np.ndarray = None,
    max_iters: int = 20,
    tol_step: float = 1e-8,
    tol_res: float = 1e-8,
) -> np.ndarray:
    # Levenberg-Marquardt FK solver with anisotropic regularization.
    rho = rho0.copy()
    if reg_weights is None:
        reg_weights = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    w = np.diag(reg_weights)
    for _ in range(max_iters):
        f, j = fk_residual_and_jacobian(rho, lengths, geom)
        lhs = j.T @ j + damping * w
        rhs = j.T @ f
        step = np.linalg.solve(lhs, rhs)
        rho_new = rho - step
        rho_new[3:] = wrap_euler(rho_new[3:])
        if np.linalg.norm(step) < tol_step or np.linalg.norm(f) < tol_res:
            rho = rho_new
            break
        rho = rho_new
    return rho


def forward_kinematics_lm_with_prior(
    rho0: np.ndarray,
    lengths: np.ndarray,
    geom: CDPRGeometry,
    rho_prior: np.ndarray,
    damping: float = 1e-3,
    reg_weights: np.ndarray = None,
    prior_weights: np.ndarray = None,
    max_iters: int = 20,
    tol_step: float = 1e-8,
    tol_res: float = 1e-8,
) -> np.ndarray:
    """LM FK solver with an additional quadratic prior term.

    Minimize: ||f(rho)||^2 + ||W_p (rho - rho_prior)||^2
    where f is cable-length residual. This helps keep the solver
    on the desired local branch and prevents sudden attitude jumps.
    """
    rho = rho0.copy()
    if reg_weights is None:
        reg_weights = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    if prior_weights is None:
        # Weak on position, stronger on attitude continuity.
        prior_weights = np.array([2.0, 2.0, 2.0, 40.0, 40.0, 40.0])
    w = np.diag(reg_weights)
    wp = np.diag(prior_weights)

    for _ in range(max_iters):
        f, j = fk_residual_and_jacobian(rho, lengths, geom)
        e_prior = rho - rho_prior
        e_prior[3:] = np.array([wrap_angle(v) for v in e_prior[3:]])
        lhs = j.T @ j + wp.T @ wp + damping * w
        rhs = j.T @ f + wp.T @ wp @ e_prior
        step = np.linalg.solve(lhs, rhs)
        rho_new = rho - step
        rho_new[3:] = wrap_euler(rho_new[3:])
        if np.linalg.norm(step) < tol_step or np.linalg.norm(f) < tol_res:
            rho = rho_new
            break
        rho = rho_new
    return rho


class EulerEKFCDPR:
    # State x = [r(3), v(3), theta(3), b_a(3), b_g(3)] -> 15x1
    def __init__(self, dt: float, g_a: np.ndarray):
        self.dt = dt
        self.g_a = g_a

        self.x = np.zeros(15)
        self.p = np.eye(15) * 1e-2

        # Process noise for [w_a, w_g, w_ba, w_bg]
        self.q = np.diag(
            np.hstack(
                [
                    np.full(3, 0.097**2),
                    np.full(3, np.deg2rad(0.56) ** 2),
                    np.full(3, 2e-4**2),
                    np.full(3, np.deg2rad(0.02) ** 2),
                ]
            )
        )

        # Measurement y = [r_fk(3), theta_fk(3)]
        self.r = np.diag(
            np.hstack(
                [
                    np.full(3, 0.002**2),
                    np.full(3, np.deg2rad(0.1) ** 2),
                ]
            )
        )

    def set_initial(self, x0: np.ndarray, p0: np.ndarray) -> None:
        self.x = x0.copy()
        self.p = p0.copy()

    def _f(self, x: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        # Discrete process model (paper Section III-B):
        # r_k   = r_{k-1} + T v_{k-1}
        # v_k   = v_{k-1} + T (C_ab (u1-b_a) + g)
        # th_k  = th_{k-1} + T S_theta^{-1}(u2-b_g)
        # bias states are random-walk in covariance model (here mean stays same)
        dt = self.dt
        r = x[0:3]
        v = x[3:6]
        th = x[6:9]
        b1 = x[9:12]
        b2 = x[12:15]

        c_ba = euler321_to_c_ba(th)
        c_ab = c_ba.T
        s_inv = np.linalg.inv(s_theta(th))

        r_n = r + dt * v
        v_n = v + dt * (c_ab @ (u1 - b1) + self.g_a)
        th_n = th + dt * (s_inv @ (u2 - b2))
        th_n = wrap_euler(th_n)
        b1_n = b1
        b2_n = b2

        return np.hstack([r_n, v_n, th_n, b1_n, b2_n])

    def _A(self, x: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        # Numerical Jacobian of f wrt state x.
        # Easier to maintain in a demo than carrying full analytic A.
        n = x.size
        eps = 1e-6
        a = np.zeros((n, n))
        fx = self._f(x, u1, u2)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            fp = self._f(x + dx, u1, u2)
            fm = self._f(x - dx, u1, u2)
            a[:, i] = (fp - fm) / (2.0 * eps)
        # Keep wrapping-consistent around branch cut.
        a[6:9, :] = np.nan_to_num(a[6:9, :], nan=0.0, posinf=0.0, neginf=0.0)
        _ = fx
        return a

    def _L(self, x: np.ndarray) -> np.ndarray:
        # wk = [w_a(3), w_g(3), w_ba(3), w_bg(3)]
        dt = self.dt
        th = x[6:9]
        c_ab = euler321_to_c_ba(th).T
        s_inv = np.linalg.inv(s_theta(th))

        l = np.zeros((15, 12))
        l[3:6, 0:3] = dt * c_ab
        l[6:9, 3:6] = dt * s_inv
        l[9:12, 6:9] = -np.eye(3)
        l[12:15, 9:12] = -np.eye(3)
        return l

    def predict(self, u1: np.ndarray, u2: np.ndarray) -> None:
        # EKF prediction.
        a = self._A(self.x, u1, u2)
        l = self._L(self.x)
        self.x = self._f(self.x, u1, u2)
        self.x[6:9] = wrap_euler(self.x[6:9])
        self.p = a @ self.p @ a.T + l @ self.q @ l.T

    def update_with_fk(self, y_fk: np.ndarray) -> None:
        # y_fk = [r_fk, theta_fk]
        # Measurement is FK pose solved from cable lengths.
        h = np.zeros((6, 15))
        h[0:3, 0:3] = np.eye(3)
        h[3:6, 6:9] = np.eye(3)

        y_pred = h @ self.x
        innovation = y_fk - y_pred
        innovation[3:6] = np.array([wrap_angle(v) for v in innovation[3:6]])

        s = h @ self.p @ h.T + self.r
        k = self.p @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ innovation
        self.x[6:9] = wrap_euler(self.x[6:9])

        i = np.eye(15)
        # Joseph-form covariance update for numerical PSD robustness.
        self.p = (i - k @ h) @ self.p @ (i - k @ h).T + k @ self.r @ k.T


def cdpr_geometry_from_calibration_file(
    calibration_file: str,
    *,
    base_dir: Optional[Path] = None,
) -> CDPRGeometry:
    """Build ``CDPRGeometry`` from kinematic calibration JSON (``a``, ``b`` keys).

    Does not construct ``CDPR`` (no ROS publishers). Relative paths resolve like
    ``cdpr.CDPR.load_kinematic_calibration``: non-absolute paths are under
    ``base_dir`` (default: directory of this module).
    """
    path = Path(calibration_file).expanduser()
    if not path.is_absolute():
        root = base_dir if base_dir is not None else Path(__file__).resolve().parent
        path = root / path
    with path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    a = np.asarray(calib["a"], dtype=float).reshape(8, 3)
    b = np.asarray(calib["b"], dtype=float).reshape(8, 3)
    return CDPRGeometry(winches_a=a.copy(), attachments_b=b.copy())


_NOMINAL_WINCHES_A = np.array(
    [
        [-0.260, -0.243, 2.300],
        [-0.361, -0.125, 2.300],
        [-2.049, -0.089, 2.300],
        [-2.169, -0.212, 2.300],
        [-2.193, -1.225, 2.290],
        [-2.084, -1.357, 2.300],
        [-0.415, -1.384, 2.300],
        [-0.290, -1.252, 2.300],
    ]
)
_NOMINAL_ATTACHMENTS_B = np.array(
    [
        [0.184, -0.125, 0.110],
        [-0.140, 0.169, -0.110],
        [0.140, 0.169, 0.110],
        [-0.184, -0.125, -0.110],
        [-0.184, 0.125, 0.110],
        [0.140, -0.169, -0.110],
        [-0.140, -0.169, 0.110],
        [0.184, 0.125, -0.110],
    ]
)


def make_demo_geometry(use_ros_cdpr: bool = False) -> CDPRGeometry:
    """Nominal CDPR geometry for FK/EKF demos and plotting.

    By default this does **not** construct ``CDPR()`` (which registers ROS
    publishers). Pass ``use_ros_cdpr=True`` only when you intentionally want
    attachment matrices from a live ``CDPR`` instance.
    """
    if use_ros_cdpr:
        try:
            from cdpr import CDPR

            cdpr = CDPR(imu_active=False)
            winches, att = cdpr.get_cable_attachment_points()
            return CDPRGeometry(winches_a=winches, attachments_b=att)
        except Exception:
            pass
    return CDPRGeometry(
        winches_a=_NOMINAL_WINCHES_A.copy(),
        attachments_b=_NOMINAL_ATTACHMENTS_B.copy(),
    )


def run_demo():
    np.random.seed(7)
    geom = make_demo_geometry()

    dt = 0.01
    steps = 1200
    t = np.arange(steps) * dt
    g_a = np.array([0.0, 0.0, -9.81])

    # Ground-truth trajectory for synthetic data generation.
    r_true = np.column_stack(
        [
            -1.25 + 0.06 * np.sin(0.6 * t),
            -0.74 + 0.06 * np.cos(0.4 * t),
            1.55 + 0.04 * np.sin(0.5 * t),
        ]
    )
    v_true = np.gradient(r_true, dt, axis=0)
    a_true = np.gradient(v_true, dt, axis=0)

    th_true = np.column_stack(
        [
            np.deg2rad(7.0) * np.sin(0.7 * t),
            np.deg2rad(6.0) * np.sin(0.5 * t + 0.5),
            np.deg2rad(10.0) * np.sin(0.45 * t),
        ]
    )
    th_dot_true = np.gradient(th_true, dt, axis=0)

    omega_true = np.zeros_like(th_true)
    for k in range(steps):
        omega_true[k] = s_theta(th_true[k]) @ th_dot_true[k]

    # Sensor models: IMU + encoder noise/bias.
    b_a0 = np.array([0.00, -0.01, 0.03])
    b_g0 = np.deg2rad(np.array([0.15, -0.12, 0.08]))
    b_a0 = np.array([0.00, -0.00, 0.00])
    b_g0 = np.deg2rad(np.array([0.0, -0.0, 0.0]))
    sig_a = 0.008
    sig_g = np.deg2rad(0.07)
    sig_ba_rw = 2e-5
    sig_bg_rw = np.deg2rad(8e-5)
    sig_len = 0.0015

    u1 = np.zeros((steps, 3))
    u2 = np.zeros((steps, 3))
    lengths_meas = np.zeros((steps, geom.m))
    b_a_true = np.zeros((steps, 3))
    b_g_true = np.zeros((steps, 3))
    b_a_true[0] = b_a0
    b_g_true[0] = b_g0
    for k in range(steps):
        if k > 0:
            b_a_true[k] = b_a_true[k - 1] + np.random.randn(3) * sig_ba_rw
            b_g_true[k] = b_g_true[k - 1] + np.random.randn(3) * sig_bg_rw
        c_ba = euler321_to_c_ba(th_true[k])
        # u1 = C_ba (a - g) + b_a + noise
        u1[k] = c_ba @ (a_true[k] - g_a) + b_a_true[k] + np.random.randn(3) * sig_a
        u2[k] = omega_true[k] + b_g_true[k] + np.random.randn(3) * sig_g
        lengths_true = cable_lengths_from_pose(r_true[k], th_true[k], geom)
        lengths_meas[k] = lengths_true + np.random.randn(geom.m) * sig_len

    ekf = EulerEKFCDPR(dt=dt, g_a=g_a)
    x0 = np.zeros(15)
    x0[0:3] = r_true[0]# + np.array([0.02, -0.015, 0.01])
    x0[6:9] = th_true[0]# + np.deg2rad(np.array([4.0, -3.0, 5.0]))
    p0 = np.eye(15) * 0.03
    p0[6:9, 6:9] *= 4.0
    ekf.set_initial(x0, p0)

    xhat = np.zeros((steps, 15))
    xhat[0] = ekf.x
    rho_fk_seed = np.hstack([x0[0:3], x0[6:9]])

    for k in range(1, steps):
        ekf.predict(u1[k - 1], u2[k - 1])
        rho_fk = forward_kinematics_lm(rho_fk_seed, lengths_meas[k], geom, max_iters=4)
        rho_fk_seed = rho_fk.copy()
        ekf.update_with_fk(rho_fk)

        xhat[k] = ekf.x

    pos_err = xhat[:, 0:3] - r_true
    att_err = xhat[:, 6:9] - th_true
    att_err = np.array([[wrap_angle(v) for v in row] for row in att_err])

    rmse_pos = np.sqrt(np.mean(pos_err**2, axis=0))
    rmse_att_deg = np.rad2deg(np.sqrt(np.mean(att_err**2, axis=0)))

    print("Euler-EKF demo finished.")
    print(f"Position RMSE [m] : x={rmse_pos[0]:.4f}, y={rmse_pos[1]:.4f}, z={rmse_pos[2]:.4f}")
    print(
        "Attitude RMSE [deg]: "
        f"roll={rmse_att_deg[0]:.3f}, pitch={rmse_att_deg[1]:.3f}, yaw={rmse_att_deg[2]:.3f}"
    )

    # Plot truth vs estimate for position and Euler angles.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    labels_pos = ["x [m]", "y [m]", "z [m]"]
    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, r_true[:, i], "k-", linewidth=1.6, label="true")
        ax.plot(t, xhat[:, i], "r--", linewidth=1.2, label="estimated")
        ax.set_ylabel(labels_pos[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

    th_true_deg = np.rad2deg(th_true)
    th_hat_deg = np.rad2deg(xhat[:, 6:9])
    labels_att = ["roll [deg]", "pitch [deg]", "yaw [deg]"]
    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, th_true_deg[:, i], "k-", linewidth=1.6, label="true")
        ax.plot(t, th_hat_deg[:, i], "r--", linewidth=1.2, label="estimated")
        ax.set_ylabel(labels_att[i])
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)

    fig.suptitle("CDPR Euler-EKF: Truth vs Estimated States")
    fig.tight_layout()

    fig_path = "cdpr_euler_ekf_truth_vs_estimate.png"
    fig.savefig(fig_path, dpi=160)
    print(f"Saved plot to: {fig_path}")
    plt.show()


if __name__ == "__main__":
    run_demo()
