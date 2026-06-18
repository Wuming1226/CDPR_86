# CDPR 模块拆分设计

**日期:** 2026-06-15  
**状态:** 已批准（brainstorming 完成）  
**目标:** 去掉 `publish_cable_lengths` / `subscribe_motor_pos` 等开关，用场景化工厂组合 ROS 组件，避免多实例重复发布 `/cable_lengths_measure`；统一几何为 `RuntimeGeometry`（废弃 `CDPRGeometry`）。

---

## 1. 背景与问题

当前 `scripts/cdpr.py` 中单个 `CDPR` 类同时承担：

- 运动学几何与绳长公式
- 动捕 / IMU 订阅与缓存
- 电机编码器 → `/cable_lengths_measure`
- `/motor_velo` 发布
- 启动时 l0 / `init_motor_pos` 初始化

多个脚本（EKF、`joystick_control`、`track`、`plot_cable_pose_vs_encoder`）各自 `CDPR()` 时，会出现多个 `/cable_lengths_measure` publisher、不同的 l0 初始化策略，以及依赖布尔开关才能避免冲突。

---

## 2. 设计原则

1. **场景化工厂**：仅通过 `CDPR.for_ekf` / `CDPR.for_velocity_control` / `CDPR.for_encoder_plot` 创建实例；**删除**旧式 `CDPR(...)` 胖构造函数（F1）。
2. **无对外布尔开关**：是否订 motor、是否发 cable 由工厂内部决定，不暴露 `publish_cable_lengths` / `subscribe_motor_pos`。
3. **几何单一来源**：`RuntimeGeometry` 定义在 `cdpr.py`，替代 `cdpr_euler_ekf.CDPRGeometry`；EKF 使用 `cdpr.geom` 同一份对象。
4. **ROS 用类、纯计算用函数**：带 Subscriber/Publisher 的块用类；绳长公式等为模块级函数。
5. **严格启动策略**：去掉 `use_identity_on_timeout`；限时轮询（默认 **10s**），超时 **`RuntimeError`**；禁止无限阻塞等待 motor。

---

## 3. 文件与模块布局

首版 **不拆多文件**，主要改动：

| 文件 | 变更 |
|------|------|
| `scripts/cdpr.py` | 新结构：`RuntimeGeometry`、ROS 类、工厂、`CDPR` 薄壳 |
| `scripts/cdpr_euler_ekf.py` | `CDPRGeometry` → `from cdpr import RuntimeGeometry`；删除/合并 loader |
| `scripts/cdpr_euler_ekf_ros_node.py` | `CDPR.for_ekf`；`self.geom = cdpr.geom` |
| `scripts/joystick_control.py` | `CDPR.for_velocity_control` |
| `scripts/track.py` | 同上 |
| `scripts/plot_cable_pose_vs_encoder.py` | `CDPR.for_encoder_plot` |
| `launch/joystick_control.launch` | 删除 cable/motor 开关 param |
| `scripts/compare_plot.py` | 几何加载改用 `load_runtime_geometry` |
| `scripts/record_fk_imu_video.py` | metadata 几何来源同上 |
| `scripts/fk_static_test.py` | `RuntimeGeometry` |

**不改动：** `compare_plot` 仍不实例化 `CDPR`；`remote_motor_ekf_bootstrap` 行为不变。

---

## 4. 数据结构与纯函数（`cdpr.py`）

### 4.1 `RuntimeGeometry`（`@dataclass` 或等价）

字段：

- `a_matrix`, `b_matrix`：`(8, 3)` 锚点
- `cable_radii`：`(8,)`
- `init_cable_lens`, `init_motor_pos`：`(8,)`，启动时写入后只读使用
- `calibration_file`：可选，日志用

**兼容 FK 类型注解**（属性别名，避免大改 `cdpr_euler_ekf.py`）：

- `winches_a` → `a_matrix`
- `attachments_b` → `b_matrix`
- `m` → `8`

### 4.2 纯函数

- `quat_valid(q) -> bool`
- `load_runtime_geometry(is_calibrated, calibration_file, use_calibrated_cable_length, base_dir=...) -> RuntimeGeometry`
- `cable_length_at_pose(geom, pos, rot) -> ndarray[8]`
- `cable_length_from_motor(geom, motor_pos) -> ndarray[8]`
- `wait_for_valid_mocap_pose(get_pose_fn, timeout=10.0) -> (x,y,z,quat)`  
  轮询直至 `quat_valid` 或超时 `RuntimeError`（**无** identity 回退）

默认超时 **10s**；可通过私有 rospy param 覆盖（实现时统一命名，例如 `~cdpr_wait_timeout` 或分项 `~mocap_wait_timeout` 等，默认均为 10）。

---

## 5. ROS 组件类（`cdpr.py`）

### 5.1 `MocapPoseCache`

- 订阅 `/vrpn_client_node/cdpr/pose`
- 保留无效大坐标过滤
- `get_pose() -> (x, y, z, quat_xyzw)`（仅 mocap，不含 IMU）
- `wait_valid_pose(timeout=10.0)` → 内部 `wait_for_valid_mocap_pose(self.get_pose, ...)`

### 5.2 `ImuOrientationCache`（可选）

- 订阅 `imu_topic`；外参加载与现 `CDPR` 一致
- `get_quat() -> ndarray | None`
- 启动时若 `imu_active=True`：在 timeout 内必须收到首包，否则 **`RuntimeError`**

### 5.3 `MotorCableBridge`

构造参数（模块内部）：

- `geom: RuntimeGeometry`
- `mocap: MocapPoseCache`
- `publish_cable: bool` — `for_ekf` 为 True；`for_encoder_plot` 为 False

行为：

- 订阅 `~motor_pos_topic`（默认 `motor_pos_abs`）
- 回调更新 `motor_pos`；若 `publish_cable` 则发布 `cable_lengths_measure`（stamp 继承 motor）
- `initialize(timeout=10.0)`：
  - 若 json 已填 `l0` / `init_motor_pos_abs`：跳过 mocap/motor 初始化
  - 否则：`mocap.wait_valid_pose` → 计算 `init_cable_lens`；限时等首包 motor → `init_motor_pos`，超时 **`RuntimeError`**

### 5.4 `MotorVelocityPublisher`

- 发布 `motor_velo`；`set_motor_velo(8,)`
- 仅 `for_velocity_control` 创建

---

## 6. 薄壳 `CDPR`

**禁止**公开胖 `__init__(...)`。仅三种工厂：

### 6.1 `CDPR.for_ekf(...)`

**用于:** `cdpr_euler_ekf_ros_node.py`

**创建:** `RuntimeGeometry` + `MocapPoseCache` +（`imu_active` 时）`ImuOrientationCache` + `MotorCableBridge(publish_cable=True)`

**不创建:** `MotorVelocityPublisher`

**参数（与现 EKF launch 对齐）:**  
`imu_active`, `imu_topic`, `is_calibrated`, `use_calibrated_cable_length`, `calibration_file`, `imu_extrinsic*`, `apply_imu_extrinsic`

**EKF 几何:** `self.geom = self.cdpr.geom`（删除 `CDPRGeometry(winches_a=wa, ...)`）

### 6.2 `CDPR.for_velocity_control(...)`

**用于:** `joystick_control.py`, `track.py`

**创建:** `RuntimeGeometry` + `MocapPoseCache` +（`imu_active` 时）`ImuOrientationCache` + `MotorVelocityPublisher`

**不创建:** `MotorCableBridge`

**参数:**

| 参数 | 默认 |
|------|------|
| `is_calibrated` | `True` |
| `calibration_file` | `cdpr_kinematic_calib.json` |
| `use_calibrated_cable_length` | 随 `is_calibrated`（True 时 l0 从 json） |
| `imu_active` | `False` |
| `imu_topic`, `imu_extrinsic*` | 与 EKF 相同语义 |

`imu_active=True` 时：`get_moving_platform_pose_from_mocap()` 位置用 mocap，姿态用 IMU。

### 6.3 `CDPR.for_encoder_plot(...)`

**用于:** `plot_cable_pose_vs_encoder.py`

**创建:** `RuntimeGeometry` + `MocapPoseCache` + `MotorCableBridge(publish_cable=False)`  
可选 `ImuOrientationCache`（`~imu_active`）

**默认几何:** `is_calibrated=False`，`use_calibrated_cable_length=False`（与现 `CDPR()` 默认一致，mocap+motor 初始化 l0）

### 6.4 转发 API（兼容现有 call site）

| 方法/属性 | 转发目标 |
|-----------|----------|
| `geom`, `a_matrix`, `b_matrix` | `RuntimeGeometry` |
| `get_moving_platform_pose_from_mocap()` | mocap 位置 +（有 imu 则）imu 姿态 |
| `wait_for_valid_mocap_pose(timeout=10)` | `MocapPoseCache.wait_valid_pose` |
| `set_motor_velo` | `MotorVelocityPublisher` |
| `calculate_cable_length_from_motor_pos` | `cable_length_from_motor(geom, motor_pos)` |
| `get_cable_attachment_points()` | 可删除并改 call site，或短期保留 deprecated 转发 |
| `motor_pos` | `MotorCableBridge.motor_pos`（无 bridge 时勿依赖） |

---

## 7. 多进程 ROS 行为（§C）

| Topic | 发布者 | 同时跑 EKF + joystick 时 |
|-------|--------|---------------------------|
| `/cable_lengths_measure` | 仅 EKF 进程 `for_ekf` | **1** publisher |
| `/motor_velo` | 仅 velocity 控制进程 | **1** publisher（单控制节点时） |

多进程重复订阅 mocap / imu / motor（EKF + encoder plot）**保持现状**，不在本轮消除。

---

## 8. 错误处理（§D）

- **删除** `use_identity_on_timeout` 及 EKF identity 姿态回退。
- **策略 A**：限时轮询，超时 **`RuntimeError`**。
- **默认 timeout：10s**（mocap 有效四元数、motor 首包、`imu_active=True` 时 IMU 首包）。
- **删除** motor 无限 `while not received` 循环。
- `imu_active=True` 时 IMU 为硬要求，超时失败，不回退 mocap 姿态。

---

## 9. 删除项（§E + F1）

- `cdpr_euler_ekf.CDPRGeometry` 类
- `cdpr_geometry_from_calibration_file`（合并为 `load_runtime_geometry` 或薄包装）
- `publish_cable_lengths`, `subscribe_motor_pos` 及 launch 中对应 param
- 旧 `CDPR(...)` 构造函数
- `cdpr_euler_ekf_ros_node._wait_valid_mocap_init_pose` 的 identity 分支（改为统一 wait + raise）
- `make_demo_geometry(use_ros_cdpr=True)` 中 live `CDPR()` 分支（改用 `load_runtime_geometry` 或 nominal）

---

## 10. 验收标准

1. EKF + `joystick_control` 同时 launch：`rostopic info /cable_lengths_measure` 仅 **1** publisher。
2. joystick/track 启动无零四元数崩溃；mocap 未就绪时在 **10s** 内失败并报错。
3. `plot_cable_pose_vs_encoder` 不增加 cable publisher，对比图正常。
4. EKF `use_calibrated_cable_length=false` 仍从 mocap+motor 初始化 l0（在 10s 内完成或失败）。
5. `grep publish_cable_lengths` 在 `cdpr_86_host` 为 0。
6. `cdpr_euler_ekf.py` 中 FK 函数使用 `RuntimeGeometry`（经别名兼容 `winches_a` / `attachments_b`）。
7. `python3 -m py_compile` 通过相关脚本。

---

## 11. 已确认的 brainstorming 决策摘要

| 议题 | 决定 |
|------|------|
| 电机桥部署 | 同进程组合，非独立 ROS 节点 |
| 组装方式 | 薄壳 `CDPR` + 工厂（A） |
| 实现风格 | ROS 用类，几何用函数 + `RuntimeGeometry` |
| `RuntimeGeometry` 位置 | **`cdpr.py`（B）**，`cdpr_euler_ekf` import 之 |
| 控制工厂名 | `for_velocity_control`（非 `for_control`） |
| 控制可选 IMU | `imu_active` 默认 False |
| 废弃 `CDPRGeometry` | 是，统一 `RuntimeGeometry` |
| 旧构造函数 | **F1 删除** |
| 超时 | 10s，失败 raise，无 identity |

---

## 12. 下一步

1. 用户审阅本 spec。  
2. 使用 **writing-plans** 编写 `docs/superpowers/plans/2026-06-15-cdpr-split.md`。  
3. 实现（subagent-driven 或 inline execution）。
