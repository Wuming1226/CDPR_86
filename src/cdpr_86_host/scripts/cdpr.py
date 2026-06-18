#! /usr/bin/env python3

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from cdpr_86_msgs.msg import CableLengthsStamped, MotorPositionsStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from imu_extrinsic import ImuExtrinsic, load_extrinsic_for_node

DEFAULT_WAIT_TIMEOUT_S = 10.0

NOMINAL_WINCHES_A = np.array(
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
NOMINAL_ATTACHMENTS_B = np.array(
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
DEFAULT_CABLE_RADII = np.full(8, 0.025, dtype=float)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _wait_timeout(default: float = DEFAULT_WAIT_TIMEOUT_S) -> float:
    try:
        return float(rospy.get_param("~cdpr_wait_timeout", default))
    except rospy.ROSException:
        return default


def quat_valid(q) -> bool:
    q = np.asarray(q, dtype=float).reshape(4)
    return bool(np.linalg.norm(q) > 1e-9 and np.isfinite(q).all())


def wait_for_valid_mocap_pose(
    get_pose_fn: Callable[[], Tuple[float, float, float, object]],
    timeout: float = DEFAULT_WAIT_TIMEOUT_S,
) -> Tuple[float, float, float, np.ndarray]:
    rate = rospy.Rate(20.0)
    deadline = rospy.Time.now().to_sec() + float(timeout)
    while not rospy.is_shutdown():
        x, y, z, quat = get_pose_fn()
        q = np.asarray(quat, dtype=float).reshape(4)
        if quat_valid(q):
            rospy.loginfo("CDPR: valid mocap pose received.")
            return float(x), float(y), float(z), q
        if rospy.Time.now().to_sec() > deadline:
            raise RuntimeError(
                f"No valid mocap quaternion within {timeout:.1f} s "
                "(VRPN may be down or not streaming)."
            )
        rate.sleep()
    raise RuntimeError("Shutdown before valid mocap pose.")


@dataclass
class RuntimeGeometry:
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    cable_radii: np.ndarray
    init_cable_lens: np.ndarray
    init_motor_pos: np.ndarray
    calibration_file: Optional[str] = None

    @property
    def winches_a(self) -> np.ndarray:
        return self.a_matrix

    @property
    def attachments_b(self) -> np.ndarray:
        return self.b_matrix

    @property
    def m(self) -> int:
        return int(self.a_matrix.shape[0])


def _resolve_calibration_path(calibration_file: str, base_dir: Optional[Path]) -> Path:
    calibration_path = Path(calibration_file).expanduser()
    if not calibration_path.is_absolute():
        root = base_dir if base_dir is not None else Path(__file__).resolve().parent
        calibration_path = root / calibration_path
    return calibration_path


def _parse_calibration_json(calib: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(calib["a"], dtype=float).reshape(8, 3)
    b = np.asarray(calib["b"], dtype=float).reshape(8, 3)
    if "r" in calib:
        radii = np.asarray(calib["r"], dtype=float).reshape(8)
    elif "radius" in calib:
        radius = np.asarray(calib["radius"], dtype=float).reshape(-1)
        if radius.size == 1:
            radii = np.full(8, float(radius[0]), dtype=float)
        elif radius.size == 8:
            radii = radius.astype(float)
        else:
            raise ValueError(f"calibration radius should have 1 or 8 values, got {radius.size}")
    else:
        radii = DEFAULT_CABLE_RADII.copy()
    return a, b, radii


def load_runtime_geometry(
    *,
    is_calibrated: bool = False,
    calibration_file: Optional[str] = None,
    use_calibrated_cable_length: Optional[bool] = None,
    base_dir: Optional[Path] = None,
) -> RuntimeGeometry:
    if use_calibrated_cable_length is None:
        use_calibrated_cable_length = is_calibrated

    a = NOMINAL_WINCHES_A.copy()
    b = NOMINAL_ATTACHMENTS_B.copy()
    radii = DEFAULT_CABLE_RADII.copy()
    init_lens = np.zeros(8, dtype=float)
    init_motor = np.zeros(8, dtype=float)
    cal_path_str: Optional[str] = None
    calib_dict = None

    if is_calibrated:
        if calibration_file is None:
            calibration_file = "cdpr_kinematic_calib.json"
        cal_path = _resolve_calibration_path(calibration_file, base_dir)
        with cal_path.open("r", encoding="utf-8") as f:
            calib_dict = json.load(f)
        a, b, radii = _parse_calibration_json(calib_dict)
        cal_path_str = str(cal_path)
        rospy.loginfo("Loaded CDPR kinematic calibration from %s", cal_path)

    if is_calibrated and use_calibrated_cable_length and calib_dict is not None:
        init_lens = np.asarray(calib_dict["l0"], dtype=float).reshape(8)
        init_motor = np.asarray(calib_dict["init_motor_pos_abs"], dtype=float).reshape(8)
        rospy.loginfo(
            "Using calibrated cable lengths (l0, init_motor_pos_abs) from %s.",
            cal_path_str,
        )

    return RuntimeGeometry(
        a_matrix=a,
        b_matrix=b,
        cable_radii=radii,
        init_cable_lens=init_lens,
        init_motor_pos=init_motor,
        calibration_file=cal_path_str,
    )


def init_cable_lens_from_mocap(geom: RuntimeGeometry, pos, quat) -> np.ndarray:
    pos0 = np.asarray(pos, dtype=float).reshape(3)
    rot0 = R.from_quat(np.asarray(quat, dtype=float).reshape(4))
    b_world = rot0.apply(geom.b_matrix)
    a = geom.a_matrix
    lens = np.array(
        [np.linalg.norm(pos0 - a[i] + b_world[i]) for i in range(8)],
        dtype=float,
    )
    rospy.loginfo("init_cable_lens: %s", lens)
    return lens


def cable_length_from_motor(geom: RuntimeGeometry, motor_pos: np.ndarray) -> np.ndarray:
    motor_pos = np.asarray(motor_pos, dtype=float).reshape(-1)
    cable_lengths = geom.init_cable_lens.copy()
    for i in range(min(len(cable_lengths), len(motor_pos))):
        motor_delta = (motor_pos[i] - geom.init_motor_pos[i]) / 10000.0 * 2.0 * np.pi
        if not i % 2:
            cable_lengths[i] = geom.init_cable_lens[i] - motor_delta * geom.cable_radii[i]
        else:
            cable_lengths[i] = geom.init_cable_lens[i] + motor_delta * geom.cable_radii[i]
    return cable_lengths


def cable_length_at_pose(geom: RuntimeGeometry, pos, rot) -> np.ndarray:
    pos0 = np.asarray(pos, dtype=float).reshape(3)
    b_matrix = rot.apply(geom.b_matrix)
    a = geom.a_matrix
    return np.array(
        [np.linalg.norm(pos0 - a[i] + b_matrix[i]) for i in range(8)],
        dtype=float,
    )


class MocapPoseCache:
    MOCAP_TOPIC = "/vrpn_client_node/cdpr/pose"

    def __init__(self) -> None:
        self._pose = PoseStamped()
        rospy.Subscriber(self.MOCAP_TOPIC, PoseStamped, self._callback, queue_size=1)

    def _callback(self, data: PoseStamped) -> None:
        if (
            np.abs(data.pose.position.x) > 2000
            or np.abs(data.pose.position.y) > 2000
            or np.abs(data.pose.position.z) > 2000
        ):
            return
        self._pose.pose.position = data.pose.position
        self._pose.pose.orientation = data.pose.orientation
        self._pose.header.frame_id = data.header.frame_id
        self._pose.header.stamp = data.header.stamp

    def get_pose(self) -> Tuple[float, float, float, list]:
        x = self._pose.pose.position.x
        y = self._pose.pose.position.y
        z = self._pose.pose.position.z
        quat = [
            self._pose.pose.orientation.x,
            self._pose.pose.orientation.y,
            self._pose.pose.orientation.z,
            self._pose.pose.orientation.w,
        ]
        return float(x), float(y), float(z), quat

    def wait_valid_pose(self, timeout: float = DEFAULT_WAIT_TIMEOUT_S):
        return wait_for_valid_mocap_pose(self.get_pose, timeout=timeout)


class ImuOrientationCache:
    def __init__(
        self,
        imu_topic: str = "/imu",
        imu_extrinsic: Optional[ImuExtrinsic] = None,
        apply_extrinsic: bool = True,
        extrinsic_file: Optional[str] = None,
        wait_timeout: float = DEFAULT_WAIT_TIMEOUT_S,
    ) -> None:
        self.imu_topic = imu_topic
        self._quat: Optional[np.ndarray] = None
        self._extrinsic = imu_extrinsic
        if apply_extrinsic and self._extrinsic is None:
            ext_path = extrinsic_file
            if ext_path is None:
                ext_path = rospy.get_param("~imu_extrinsic_file", "cdpr_imu_extrinsic.json")
            self._extrinsic = load_extrinsic_for_node(
                ext_path,
                enabled=True,
                node_name="CDPR",
            )
        if self._extrinsic is not None:
            rospy.loginfo(
                "CDPR IMU extrinsic loaded (n=%d, residual_rms=%.4f deg).",
                self._extrinsic.n_samples,
                self._extrinsic.residual_angle_deg_rms,
            )
        rospy.Subscriber(self.imu_topic, Imu, self._callback, queue_size=1)
        try:
            imu_msg = rospy.wait_for_message(self.imu_topic, Imu, timeout=wait_timeout)
            self._store_quat(
                np.array(
                    [
                        imu_msg.orientation.x,
                        imu_msg.orientation.y,
                        imu_msg.orientation.z,
                        imu_msg.orientation.w,
                    ],
                    dtype=float,
                )
            )
            rospy.loginfo("CDPR got first IMU message from %s.", self.imu_topic)
        except rospy.ROSException as exc:
            raise RuntimeError(
                f"No IMU message on {self.imu_topic} within {wait_timeout:.1f} s"
            ) from exc

    def _correct_quat(self, quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)
        if self._extrinsic is None:
            return q
        corrected = self._extrinsic.apply_quat(q)
        return corrected if corrected is not None else q

    def _store_quat(self, quat_xyzw: np.ndarray) -> None:
        self._quat = self._correct_quat(quat_xyzw)

    def _callback(self, data: Imu) -> None:
        self._store_quat(
            np.array(
                [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w],
                dtype=float,
            )
        )

    def get_quat(self) -> Optional[np.ndarray]:
        return self._quat


class MotorVelocityPublisher:
    def __init__(self) -> None:
        self._pub = rospy.Publisher("motor_velo", Float32MultiArray, queue_size=10)

    def set_motor_velo(self, motor_velo) -> None:
        self._pub.publish(Float32MultiArray(data=np.asarray(motor_velo, dtype=float)))


class MotorCableBridge:
    def __init__(
        self,
        geom: RuntimeGeometry,
        mocap: MocapPoseCache,
        *,
        publish_cable: bool,
    ) -> None:
        self.geom = geom
        self.mocap = mocap
        self.publish_cable = publish_cable
        self.motor_pos = np.zeros(8, dtype=float)
        self._motor_received = False
        self.motor_pos_topic = rospy.get_param("~motor_pos_topic", "motor_pos_abs")
        self._cable_pub = None
        if publish_cable:
            self._cable_pub = rospy.Publisher(
                "cable_lengths_measure", CableLengthsStamped, queue_size=50
            )
        rospy.Subscriber(
            self.motor_pos_topic,
            MotorPositionsStamped,
            self._callback,
            queue_size=1,
        )

    def _callback(self, data: MotorPositionsStamped) -> None:
        self.motor_pos = np.array(data.positions, dtype=float)
        self._motor_received = True
        if self._cable_pub is None:
            return
        cable_msg = CableLengthsStamped()
        cable_msg.header.stamp = (
            data.header.stamp if data.header.stamp != rospy.Time() else rospy.Time.now()
        )
        cable_msg.header.frame_id = data.header.frame_id or "world"
        cable_msg.lengths = cable_length_from_motor(self.geom, self.motor_pos).tolist()
        self._cable_pub.publish(cable_msg)

    def initialize(self, *, use_calibrated_cable_length: bool, timeout: float) -> None:
        if use_calibrated_cable_length:
            return
        x, y, z, quat = self.mocap.wait_valid_pose(timeout=timeout)
        self.geom.init_cable_lens = init_cable_lens_from_mocap(self.geom, [x, y, z], quat)
        self._wait_first_motor(timeout=timeout)
        self.geom.init_motor_pos = self.motor_pos.copy()
        rospy.loginfo(
            "MotorCableBridge: l0 from mocap; init_motor_pos from first %s",
            self.motor_pos_topic,
        )

    def _wait_first_motor(self, timeout: float) -> None:
        deadline = rospy.Time.now().to_sec() + float(timeout)
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown():
            if self._motor_received:
                return
            if rospy.Time.now().to_sec() > deadline:
                raise RuntimeError(
                    f"No motor_pos on {self.motor_pos_topic} within {timeout:.1f} s"
                )
            rate.sleep()
        raise RuntimeError("Shutdown before first motor_pos_abs.")


class CDPR:
    """Thin facade; construct only via for_ekf / for_velocity_control / for_encoder_plot."""

    def __init__(
        self,
        *,
        geom: RuntimeGeometry,
        mocap: MocapPoseCache,
        imu: Optional[ImuOrientationCache] = None,
        bridge: Optional[MotorCableBridge] = None,
        velo: Optional[MotorVelocityPublisher] = None,
        imu_active: bool = False,
        imu_topic: str = "/imu",
    ) -> None:
        self.geom = geom
        self._mocap = mocap
        self._imu = imu
        self._bridge = bridge
        self._velo = velo
        self.imu_active = bool(imu_active)
        self.imu_topic = imu_topic

    @property
    def a_matrix(self) -> np.ndarray:
        return self.geom.a_matrix

    @property
    def b_matrix(self) -> np.ndarray:
        return self.geom.b_matrix

    @property
    def init_cable_lens(self) -> np.ndarray:
        return self.geom.init_cable_lens

    @property
    def init_motor_pos(self) -> np.ndarray:
        return self.geom.init_motor_pos

    @property
    def cable_radii(self) -> np.ndarray:
        return self.geom.cable_radii

    @property
    def motor_pos(self) -> Optional[np.ndarray]:
        if self._bridge is None:
            return None
        return self._bridge.motor_pos

    @classmethod
    def for_ekf(
        cls,
        imu_active: bool = False,
        imu_topic: str = "/imu",
        is_calibrated: bool = False,
        use_calibrated_cable_length: Optional[bool] = None,
        calibration_file: Optional[str] = None,
        imu_extrinsic_file: Optional[str] = None,
        apply_imu_extrinsic: bool = True,
        imu_extrinsic: Optional[ImuExtrinsic] = None,
    ) -> "CDPR":
        if use_calibrated_cable_length is None:
            use_calibrated_cable_length = is_calibrated
        try:
            use_calibrated_cable_length = _as_bool(
                rospy.get_param("~use_calibrated_cable_length", use_calibrated_cable_length)
            )
        except rospy.ROSException:
            pass

        timeout = _wait_timeout()
        geom = load_runtime_geometry(
            is_calibrated=is_calibrated,
            calibration_file=calibration_file,
            use_calibrated_cable_length=use_calibrated_cable_length,
        )
        mocap = MocapPoseCache()
        imu_cache = None
        if imu_active:
            ext = imu_extrinsic
            if apply_imu_extrinsic and ext is None:
                ext_path = imu_extrinsic_file or rospy.get_param(
                    "~imu_extrinsic_file", "cdpr_imu_extrinsic.json"
                )
            else:
                ext_path = imu_extrinsic_file
            imu_cache = ImuOrientationCache(
                imu_topic=imu_topic,
                imu_extrinsic=ext if apply_imu_extrinsic else None,
                apply_extrinsic=apply_imu_extrinsic,
                extrinsic_file=ext_path if apply_imu_extrinsic and ext is None else None,
                wait_timeout=timeout,
            )
        bridge = MotorCableBridge(geom, mocap, publish_cable=True)
        bridge.initialize(
            use_calibrated_cable_length=use_calibrated_cable_length,
            timeout=timeout,
        )
        rospy.loginfo(
            "CDPR.for_ekf: mocap + motor bridge (publish cable)%s",
            " + imu" if imu_cache else "",
        )
        return cls(
            geom=geom,
            mocap=mocap,
            imu=imu_cache,
            bridge=bridge,
            velo=None,
            imu_active=imu_active,
            imu_topic=imu_topic,
        )

    @classmethod
    def for_velocity_control(
        cls,
        is_calibrated: bool = True,
        calibration_file: Optional[str] = "cdpr_kinematic_calib.json",
        use_calibrated_cable_length: Optional[bool] = None,
        imu_active: bool = False,
        imu_topic: str = "/imu",
        imu_extrinsic_file: Optional[str] = None,
        apply_imu_extrinsic: bool = True,
        imu_extrinsic: Optional[ImuExtrinsic] = None,
    ) -> "CDPR":
        if use_calibrated_cable_length is None:
            use_calibrated_cable_length = is_calibrated
        timeout = _wait_timeout()
        geom = load_runtime_geometry(
            is_calibrated=is_calibrated,
            calibration_file=calibration_file,
            use_calibrated_cable_length=use_calibrated_cable_length,
        )
        mocap = MocapPoseCache()
        imu_cache = None
        if imu_active:
            imu_cache = ImuOrientationCache(
                imu_topic=imu_topic,
                imu_extrinsic=imu_extrinsic,
                apply_extrinsic=apply_imu_extrinsic,
                extrinsic_file=imu_extrinsic_file,
                wait_timeout=timeout,
            )
        velo = MotorVelocityPublisher()
        rospy.loginfo(
            "CDPR.for_velocity_control: mocap + motor_velo (no cable bridge)%s",
            " + imu" if imu_cache else "",
        )
        return cls(
            geom=geom,
            mocap=mocap,
            imu=imu_cache,
            bridge=None,
            velo=velo,
            imu_active=imu_active,
            imu_topic=imu_topic,
        )

    @classmethod
    def for_encoder_plot(
        cls,
        imu_active: bool = False,
        imu_topic: str = "/imu",
        imu_extrinsic_file: Optional[str] = None,
        apply_imu_extrinsic: bool = True,
        imu_extrinsic: Optional[ImuExtrinsic] = None,
    ) -> "CDPR":
        timeout = _wait_timeout()
        geom = load_runtime_geometry(
            is_calibrated=False,
            use_calibrated_cable_length=False,
        )
        mocap = MocapPoseCache()
        imu_cache = None
        if imu_active:
            imu_cache = ImuOrientationCache(
                imu_topic=imu_topic,
                imu_extrinsic=imu_extrinsic,
                apply_extrinsic=apply_imu_extrinsic,
                extrinsic_file=imu_extrinsic_file,
                wait_timeout=timeout,
            )
        bridge = MotorCableBridge(geom, mocap, publish_cable=False)
        bridge.initialize(use_calibrated_cable_length=False, timeout=timeout)
        rospy.loginfo(
            "CDPR.for_encoder_plot: mocap + motor subscribe (no cable publish)%s",
            " + imu" if imu_cache else "",
        )
        return cls(
            geom=geom,
            mocap=mocap,
            imu=imu_cache,
            bridge=bridge,
            velo=None,
            imu_active=imu_active,
            imu_topic=imu_topic,
        )

    def get_moving_platform_pose_from_mocap(self):
        x, y, z, mocap_quat = self._mocap.get_pose()
        quat = mocap_quat
        if self.imu_active and self._imu is not None:
            imu_q = self._imu.get_quat()
            if imu_q is not None:
                quat = imu_q.tolist()
        return x, y, z, quat

    def wait_for_valid_mocap_pose(self, timeout: float = DEFAULT_WAIT_TIMEOUT_S):
        return self._mocap.wait_valid_pose(timeout=timeout)

    def set_motor_velo(self, motor_velo) -> None:
        if self._velo is None:
            raise RuntimeError("set_motor_velo called but MotorVelocityPublisher was not created")
        self._velo.set_motor_velo(motor_velo)

    def calculate_cable_length_from_motor_pos(self, motor_pos=None):
        if motor_pos is None:
            if self._bridge is None:
                motor_pos = self.geom.init_motor_pos
            else:
                motor_pos = self._bridge.motor_pos
        return cable_length_from_motor(self.geom, motor_pos)

    def calculate_cable_length_at_pose(self, pos, rot):
        return cable_length_at_pose(self.geom, pos, rot)

    def get_cable_attachment_points(self):
        return self.geom.a_matrix.copy(), self.geom.b_matrix.copy()


if __name__ == "__main__":
    rospy.init_node("cdpr_control", anonymous=False)
    cdpr = CDPR.for_velocity_control()
    time.sleep(1)
    print("start")
    for _ in range(1):
        cdpr.set_motor_velo([0.5] * 8)
        time.sleep(0.2)
    print("end")
