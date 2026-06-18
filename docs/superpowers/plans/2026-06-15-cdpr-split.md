# CDPR 模块拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `cdpr.py` into composable ROS components with scene factories (`for_ekf`, `for_velocity_control`, `for_encoder_plot`), replace `CDPRGeometry` with `RuntimeGeometry`, and eliminate duplicate `/cable_lengths_measure` publishers.

**Architecture:** `RuntimeGeometry` + pure cable functions live in `cdpr.py`. ROS blocks (`MocapPoseCache`, `ImuOrientationCache`, `MotorCableBridge`, `MotorVelocityPublisher`) are composed by thin `CDPR` classmethods. EKF uses `cdpr.geom` directly for FK. All readiness waits use 10s timeout then `RuntimeError` (no identity quaternion fallback).

**Tech Stack:** ROS Noetic, rospy, numpy, scipy.spatial.transform, existing `cdpr_86_msgs`, `imu_extrinsic.py`.

**Spec:** [docs/superpowers/specs/2026-06-15-cdpr-split-design.md](../specs/2026-06-15-cdpr-split-design.md)

---

## File map

| File | Responsibility after refactor |
|------|------------------------------|
| `scripts/cdpr.py` | `RuntimeGeometry`, helpers, ROS classes, `CDPR` factories |
| `scripts/cdpr_euler_ekf.py` | FK math only; `RuntimeGeometry` type hints; nominal constants; thin `load_runtime_geometry` wrapper or import |
| `scripts/cdpr_euler_ekf_ros_node.py` | `CDPR.for_ekf`, `self.geom = cdpr.geom` |
| `scripts/joystick_control.py` | `CDPR.for_velocity_control` |
| `scripts/track.py` | `CDPR.for_velocity_control` |
| `scripts/plot_cable_pose_vs_encoder.py` | `CDPR.for_encoder_plot` |
| `launch/joystick_control.launch` | Remove switch params |
| `scripts/compare_plot.py` | `load_runtime_geometry` |
| `scripts/record_fk_imu_video.py` | `load_runtime_geometry` |
| `scripts/fk_static_test.py` | `RuntimeGeometry` |

---

### Task 1: Add `RuntimeGeometry` and pure geometry functions

**Files:**
- Modify: `scripts/cdpr.py` (insert before existing `CDPR` class; keep old class until Task 4)

- [ ] **Step 1: Add imports and constants**

At top of `cdpr.py` after existing imports, add `dataclasses.dataclass`, `field`, `typing.Optional`.

Add module constants (move from current `CDPR.__init__` anchor arrays):

```python
NOMINAL_WINCHES_A = np.array([...])  # 8x3, copy from current _anchorA1..A8
NOMINAL_ATTACHMENTS_B = np.array([...])  # 8x3
DEFAULT_CABLE_RADII = np.full(8, 0.025, dtype=float)
DEFAULT_WAIT_TIMEOUT_S = 10.0
```

- [ ] **Step 2: Add `RuntimeGeometry`**

```python
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
```

- [ ] **Step 3: Add `load_runtime_geometry`**

Port logic from `load_kinematic_calibration` + nominal defaults + json `l0` / `init_motor_pos_abs` when `is_calibrated and use_calibrated_cable_length`.

Signature:

```python
def load_runtime_geometry(
    *,
    is_calibrated: bool = False,
    calibration_file: Optional[str] = None,
    use_calibrated_cable_length: Optional[bool] = None,
    base_dir: Optional[Path] = None,
) -> RuntimeGeometry:
```

Return geometry with `init_cable_lens` / `init_motor_pos` zeros if not loaded from json yet.

- [ ] **Step 4: Add pure cable functions**

```python
def cable_length_at_pose(geom: RuntimeGeometry, pos, rot) -> np.ndarray:
    # port calculate_cable_length_at_pose body

def cable_length_from_motor(geom: RuntimeGeometry, motor_pos: np.ndarray) -> np.ndarray:
    # port calculate_cable_length_from_motor_pos body using geom.init_cable_lens, etc.

def init_cable_lens_from_mocap(geom: RuntimeGeometry, pos, quat) -> np.ndarray:
    # port init_cable_length math (8 norms), return new lens array
```

- [ ] **Step 5: Refactor `wait_for_valid_mocap_pose`**

Replace method with module function (remove `use_identity_on_timeout`):

```python
def wait_for_valid_mocap_pose(get_pose_fn, timeout: float = DEFAULT_WAIT_TIMEOUT_S):
    rate = rospy.Rate(20.0)
    deadline = rospy.Time.now().to_sec() + float(timeout)
    while not rospy.is_shutdown():
        x, y, z, quat = get_pose_fn()
        if quat_valid(quat):
            return float(x), float(y), float(z), np.asarray(quat, dtype=float).reshape(4)
        if rospy.Time.now().to_sec() > deadline:
            raise RuntimeError(f"No valid mocap quaternion within {timeout:.1f} s")
        rate.sleep()
    raise RuntimeError("Shutdown before valid mocap pose")
```

- [ ] **Step 6: Verify syntax**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr.py
```

---

### Task 2: ROS component classes

**Files:**
- Modify: `scripts/cdpr.py`

- [ ] **Step 1: `MocapPoseCache`**

```python
class MocapPoseCache:
  MOCAP_TOPIC = "/vrpn_client_node/cdpr/pose"

  def __init__(self):
      self._pose = PoseStamped()
      rospy.Subscriber(self.MOCAP_TOPIC, PoseStamped, self._callback, queue_size=1)

  def _callback(self, data):  # port _pose_callback filter
  def get_pose(self) -> tuple:  # x,y,z, quat list — mocap only
  def wait_valid_pose(self, timeout=DEFAULT_WAIT_TIMEOUT_S):
      return wait_for_valid_mocap_pose(self.get_pose, timeout=timeout)
```

- [ ] **Step 2: `ImuOrientationCache`**

Port `_imu_callback`, `_correct_imu_quat`, `_store_imu_quat`, extrinsic load from old `CDPR.__init__`.

```python
class ImuOrientationCache:
  def __init__(self, imu_topic, imu_extrinsic=None, apply_extrinsic=True, extrinsic_file=None, wait_timeout=DEFAULT_WAIT_TIMEOUT_S):
      # subscribe Imu
      # if imu_active path: rospy.wait_for_message with timeout → RuntimeError on failure

  def get_quat(self) -> Optional[np.ndarray]:
```

Constructor always requires IMU (only instantiated when `imu_active=True`).

- [ ] **Step 3: `MotorVelocityPublisher`**

```python
class MotorVelocityPublisher:
  def __init__(self):
      self._pub = rospy.Publisher("motor_velo", Float32MultiArray, queue_size=10)

  def set_motor_velo(self, motor_velo):
      self._pub.publish(Float32MultiArray(data=np.asarray(motor_velo, dtype=float)))
```

- [ ] **Step 4: Verify syntax**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr.py
```

---

### Task 3: `MotorCableBridge`

**Files:**
- Modify: `scripts/cdpr.py`

- [ ] **Step 1: Implement class**

```python
class MotorCableBridge:
    def __init__(self, geom: RuntimeGeometry, mocap: MocapPoseCache, *, publish_cable: bool):
        self.geom = geom
        self.mocap = mocap
        self.publish_cable = publish_cable
        self.motor_pos = np.zeros(8, dtype=float)
        self._motor_received = False
        self.motor_pos_topic = rospy.get_param("~motor_pos_topic", "motor_pos_abs")
        if publish_cable:
            self._cable_pub = rospy.Publisher("cable_lengths_measure", CableLengthsStamped, queue_size=50)
        else:
            self._cable_pub = None
        rospy.Subscriber(self.motor_pos_topic, MotorPositionsStamped, self._callback, queue_size=1)

    def _callback(self, data: MotorPositionsStamped):
        # port _motor_pos_callback; publish only if self._cable_pub

    def initialize(self, *, use_calibrated_cable_length: bool, timeout: float = DEFAULT_WAIT_TIMEOUT_S):
        if use_calibrated_cable_length:
            return  # l0 already in geom from json
        x, y, z, quat = self.mocap.wait_valid_pose(timeout=timeout)
        self.geom.init_cable_lens = init_cable_lens_from_mocap(self.geom, [x,y,z], quat)
        self._wait_first_motor(timeout=timeout)
        self.geom.init_motor_pos = self.motor_pos.copy()
        rospy.loginfo("MotorCableBridge: l0 from mocap, init_motor_pos from first %s", self.motor_pos_topic)

    def _wait_first_motor(self, timeout: float):
        deadline = rospy.Time.now().to_sec() + timeout
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown():
            if self._motor_received:
                return
            if rospy.Time.now().to_sec() > deadline:
                raise RuntimeError(f"No motor_pos on {self.motor_pos_topic} within {timeout:.1f} s")
            rate.sleep()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr.py
```

---

### Task 4: Replace `CDPR` with thin shell + factories

**Files:**
- Modify: `scripts/cdpr.py` (delete old `CDPR.__init__` body and methods now moved)

- [ ] **Step 1: Private constructor**

```python
class CDPR:
    def __init__(self, *, geom, mocap, imu=None, bridge=None, velo=None, imu_active=False):
        self.geom = geom
        self._mocap = mocap
        self._imu = imu
        self._bridge = bridge
        self._velo = velo
        self.imu_active = bool(imu_active)
        self.imu_topic = imu.imu_topic if imu else "/imu"  # or store on factory args

    @property
    def a_matrix(self): return self.geom.a_matrix
    @property
    def b_matrix(self): return self.geom.b_matrix
    @property
    def init_cable_lens(self): return self.geom.init_cable_lens
    @property
    def init_motor_pos(self): return self.geom.init_motor_pos
    @property
    def motor_pos(self):
        if self._bridge is None:
            return None
        return self._bridge.motor_pos
```

- [ ] **Step 2: Forwarding methods**

Port `get_moving_platform_pose_from_mocap`, `wait_for_valid_mocap_pose`, `set_motor_velo`, `calculate_cable_length_from_motor_pos`, `get_cable_attachment_points`.

`set_motor_velo` raises if `_velo is None`.

- [ ] **Step 3: `for_ekf` classmethod**

```python
@classmethod
def for_ekf(cls, imu_active=False, imu_topic="/imu", is_calibrated=False,
            use_calibrated_cable_length=None, calibration_file=None,
            imu_extrinsic=None, apply_imu_extrinsic=True, imu_extrinsic_file=None):
    if use_calibrated_cable_length is None:
        use_calibrated_cable_length = is_calibrated
    geom = load_runtime_geometry(is_calibrated=is_calibrated, calibration_file=calibration_file,
                                 use_calibrated_cable_length=use_calibrated_cable_length)
    mocap = MocapPoseCache()
    imu = None
    if imu_active:
        imu = ImuOrientationCache(imu_topic, ...)
    bridge = MotorCableBridge(geom, mocap, publish_cable=True)
    bridge.initialize(use_calibrated_cable_length=use_calibrated_cable_length)
    rospy.loginfo("CDPR.for_ekf: mocap + motor bridge (publish cable)%s",
                  " + imu" if imu else "")
    return cls(geom=geom, mocap=mocap, imu=imu, bridge=bridge, velo=None, imu_active=imu_active)
```

Read `~cdpr_wait_timeout` / per-item params default 10.0 where applicable.

- [ ] **Step 4: `for_velocity_control` classmethod**

```python
@classmethod
def for_velocity_control(cls, is_calibrated=True, calibration_file="cdpr_kinematic_calib.json",
                         use_calibrated_cable_length=None, imu_active=False, imu_topic="/imu", ...):
    # geom from json l0; NO bridge
    velo = MotorVelocityPublisher()
    return cls(geom=geom, mocap=mocap, imu=imu, bridge=None, velo=velo, imu_active=imu_active)
```

- [ ] **Step 5: `for_encoder_plot` classmethod**

```python
@classmethod
def for_encoder_plot(cls, imu_active=False, imu_topic="/imu", ...):
    geom = load_runtime_geometry(is_calibrated=False, use_calibrated_cable_length=False)
    bridge = MotorCableBridge(geom, mocap, publish_cable=False)
    bridge.initialize(use_calibrated_cable_length=False)
    return cls(...)
```

- [ ] **Step 6: Update `__main__`**

Replace `cdpr = CDPR()` with `cdpr = CDPR.for_ekf()` or remove demo block.

- [ ] **Step 7: Remove dead code**

Delete old `publish_cable_lengths`, `subscribe_motor_pos`, `use_identity_on_timeout`, `load_kinematic_calibration` method (logic in `load_runtime_geometry`), duplicate methods.

- [ ] **Step 8: Verify**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr.py
grep -n "publish_cable_lengths\|subscribe_motor_pos\|use_identity" /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr.py
# Expected: no matches
```

---

### Task 5: Migrate `cdpr_euler_ekf.py` off `CDPRGeometry`

**Files:**
- Modify: `scripts/cdpr_euler_ekf.py`

- [ ] **Step 1: Replace import and type hints**

At top (careful: avoid circular import at module level for FK-only use):

Option A — type hint string only in ekf file, import RuntimeGeometry only in functions that need it.

Replace:

```python
class CDPRGeometry:
```

with re-export shim **or** delete class and add:

```python
from cdpr import RuntimeGeometry
```

Update all `geom: CDPRGeometry` → `geom: RuntimeGeometry` (13 occurrences).

- [ ] **Step 2: Replace `cdpr_geometry_from_calibration_file`**

```python
def cdpr_geometry_from_calibration_file(calibration_file, *, base_dir=None) -> RuntimeGeometry:
    from cdpr import load_runtime_geometry
    return load_runtime_geometry(
        is_calibrated=True,
        calibration_file=calibration_file,
        use_calibrated_cable_length=False,
        base_dir=base_dir,
    )
```

Keep function name for backward compat OR rename call sites to `load_runtime_geometry` (prefer rename at call sites in Task 8).

- [ ] **Step 3: Simplify `make_demo_geometry`**

Remove `use_ros_cdpr` branch entirely:

```python
def make_demo_geometry() -> RuntimeGeometry:
    from cdpr import RuntimeGeometry, NOMINAL_WINCHES_A, NOMINAL_ATTACHMENTS_B, DEFAULT_CABLE_RADII
    return RuntimeGeometry(
        a_matrix=NOMINAL_WINCHES_A.copy(),
        b_matrix=NOMINAL_ATTACHMENTS_B.copy(),
        cable_radii=DEFAULT_CABLE_RADII.copy(),
        init_cable_lens=np.zeros(8),
        init_motor_pos=np.zeros(8),
    )
```

- [ ] **Step 4: Verify**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr_euler_ekf.py
grep -n "CDPRGeometry" /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr_euler_ekf.py
# Expected: no matches (except maybe comment)
```

**Circular import note:** `cdpr.py` imports `jacobian` not `cdpr_euler_ekf`. `cdpr_euler_ekf.py` importing `RuntimeGeometry` from `cdpr` is OK if `cdpr_euler_ekf` does not get imported at `cdpr.py` module load time. Current `cdpr.py` does not import `cdpr_euler_ekf` — safe.

---

### Task 6: Update EKF ROS node

**Files:**
- Modify: `scripts/cdpr_euler_ekf_ros_node.py`

- [ ] **Step 1: Change imports**

```python
from cdpr import CDPR
# Remove CDPRGeometry from cdpr_euler_ekf import if only used for geom wrapper
```

- [ ] **Step 2: Replace construction**

```python
self.cdpr = CDPR.for_ekf(
    imu_active=self.rpy_from_imu,
    is_calibrated=self.is_calibrated,
    use_calibrated_cable_length=self.use_calibrated_cable_length,
    calibration_file=(self.calibration_file if self.is_calibrated else None),
    imu_extrinsic=self._imu_extrinsic,
    apply_imu_extrinsic=self.apply_imu_extrinsic,
    imu_extrinsic_file=self.imu_extrinsic_file,
    imu_topic=self.imu_topic,
)
self.geom = self.cdpr.geom
```

Delete lines:

```python
wa, wb = self.cdpr.get_cable_attachment_points()
self.geom = CDPRGeometry(winches_a=wa, attachments_b=wb)
```

- [ ] **Step 3: Simplify `_wait_valid_mocap_init_pose`**

```python
def _wait_valid_mocap_init_pose(self):
    timeout = float(rospy.get_param("~cdpr_wait_timeout", 10.0))
    return self.cdpr.wait_for_valid_mocap_pose(timeout=timeout)
```

Remove identity fallback and duplicate loop.

- [ ] **Step 4: Verify**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/cdpr_euler_ekf_ros_node.py
```

---

### Task 7: Update control scripts and launch

**Files:**
- Modify: `scripts/joystick_control.py`
- Modify: `scripts/track.py`
- Modify: `launch/joystick_control.launch`

- [ ] **Step 1: `joystick_control.py`**

Replace:

```python
self.cdpr = CDPR(
    imu_active=False,
    is_calibrated=True,
    calibration_file="cdpr_kinematic_calib.json",
    publish_cable_lengths=False,
    subscribe_motor_pos=False,
)
```

with:

```python
self.cdpr = CDPR.for_velocity_control(
    is_calibrated=True,
    calibration_file="cdpr_kinematic_calib.json",
    imu_active=rospy.get_param("~imu_active", False),  # optional if not in launch yet
)
```

- [ ] **Step 2: `track.py`**

Same pattern as joystick.

- [ ] **Step 3: `joystick_control.launch`**

Remove lines:

```xml
<param name="publish_cable_lengths" value="false" />
<param name="subscribe_motor_pos" value="false" />
```

- [ ] **Step 4: Verify**

```bash
python3 -m py_compile /home/xyc/CDPR_86/src/cdpr_86_host/scripts/joystick_control.py /home/xyc/CDPR_86/src/cdpr_86_host/scripts/track.py
grep -r "publish_cable_lengths\|subscribe_motor_pos" /home/xyc/CDPR_86/src/cdpr_86_host --include="*.py" --include="*.launch"
# Expected: no matches (except docs)
```

---

### Task 8: Update remaining consumers

**Files:**
- Modify: `scripts/plot_cable_pose_vs_encoder.py`
- Modify: `scripts/compare_plot.py`
- Modify: `scripts/record_fk_imu_video.py`
- Modify: `scripts/fk_static_test.py`

- [ ] **Step 1: `plot_cable_pose_vs_encoder.py`**

```python
self.cdpr = CDPR.for_encoder_plot(imu_active=imu_active, imu_topic=imu_topic)
```

- [ ] **Step 2: `compare_plot.py`**

```python
from cdpr import load_runtime_geometry
# ...
self.geom = load_runtime_geometry(
    is_calibrated=self.is_calibrated,
    calibration_file=self.calibration_file,
    use_calibrated_cable_length=False,
    base_dir=Path(__file__).resolve().parent,
)
```

Or keep using `make_demo_geometry()` when not calibrated.

- [ ] **Step 3: `record_fk_imu_video.py`**

Replace `CDPRGeometry` / `cdpr_geometry_from_calibration_file` imports with `from cdpr import RuntimeGeometry, load_runtime_geometry`.

- [ ] **Step 4: `fk_static_test.py`**

```python
from cdpr import RuntimeGeometry, NOMINAL_WINCHES_A, NOMINAL_ATTACHMENTS_B, DEFAULT_CABLE_RADII
```

Build `RuntimeGeometry` in `make_static_geometry()`.

- [ ] **Step 5: Verify all**

```bash
python3 -m py_compile \
  /home/xyc/CDPR_86/src/cdpr_86_host/scripts/plot_cable_pose_vs_encoder.py \
  /home/xyc/CDPR_86/src/cdpr_86_host/scripts/compare_plot.py \
  /home/xyc/CDPR_86/src/cdpr_86_host/scripts/record_fk_imu_video.py \
  /home/xyc/CDPR_86/src/cdpr_86_host/scripts/fk_static_test.py
```

---

### Task 9: Final verification

- [ ] **Step 1: Grep cleanup checks**

```bash
cd /home/xyc/CDPR_86/src/cdpr_86_host
grep -r "CDPRGeometry" scripts --include="*.py" | grep -v cdpr_euler_ekf.py || true
grep -r "CDPR(" scripts --include="*.py"
# CDPR( should only appear as CDPR.for_* or class definition, not CDPR(...)
grep -r "publish_cable_lengths\|subscribe_motor_pos\|use_identity" scripts launch
```

- [ ] **Step 2: Compile all touched scripts**

```bash
python3 -m py_compile scripts/cdpr.py scripts/cdpr_euler_ekf.py scripts/cdpr_euler_ekf_ros_node.py \
  scripts/joystick_control.py scripts/track.py scripts/plot_cable_pose_vs_encoder.py \
  scripts/compare_plot.py scripts/record_fk_imu_video.py scripts/fk_static_test.py
```

- [ ] **Step 3: Runtime smoke (manual, when ROS stack up)**

1. `roslaunch cdpr_86_host ekf_with_remote_motor.launch` — EKF starts, one cable publisher.
2. Add `roslaunch cdpr_86_host joystick_control.launch` — still one cable publisher.
3. Joystick moves platform; no quat crash when mocap up.

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| RuntimeGeometry in cdpr.py | Task 1 |
| Pure functions | Task 1 |
| MocapPoseCache / Imu / Bridge / Velo | Tasks 2–3 |
| for_ekf / for_velocity_control / for_encoder_plot | Task 4 |
| Delete old CDPR(...) | Task 4 |
| Remove CDPRGeometry | Task 5 |
| EKF self.geom = cdpr.geom | Task 6 |
| Remove launch switches | Task 7 |
| 10s timeout, raise, no identity | Tasks 1, 3, 4, 6 |
| plot encoder no cable pub | Tasks 3, 4, 8 |
| Acceptance grep/py_compile | Task 9 |

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-cdpr-split.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task (1→9), review between tasks.
2. **Inline Execution** — Implement Tasks 1–9 sequentially in this session with checkpoints after Task 4 and Task 6.

Which approach?
