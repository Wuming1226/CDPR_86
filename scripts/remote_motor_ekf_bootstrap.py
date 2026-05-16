#!/usr/bin/env python3
import os
import shlex
import signal
import subprocess
import sys
import time

import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


class RemoteMotorEkfBootstrap:
    def __init__(self) -> None:
        self.remote_host = rospy.get_param("~remote_host", "192.168.5.104")
        self.remote_user = rospy.get_param("~remote_user", "will")
        self.remote_port = int(rospy.get_param("~remote_port", 22))
        self.remote_process_pattern = rospy.get_param("~remote_process_pattern", "cdpr_86.py")
        self.remote_start_command = rospy.get_param("~remote_start_command", "rosrun cdpr_86_actuator cdpr_86.py")
        self.remote_setup_script = rospy.get_param("~remote_setup_script", "~/CDPR_86/devel/setup.bash")
        self.remote_log_file = rospy.get_param("~remote_log_file", "~/motor_node_bootstrap.log")
        self.motor_topic = rospy.get_param("~motor_topic", "/motor_pos_abs")
        self.motor_topic_wait_timeout = float(rospy.get_param("~motor_topic_wait_timeout", 20.0))
        self.post_restart_wait_s = float(rospy.get_param("~post_restart_wait_s", 1.0))

        self.remote2_host = rospy.get_param("~remote2_host", "192.168.5.105")
        self.remote2_user = rospy.get_param("~remote2_user", "will")
        self.remote2_port = int(rospy.get_param("~remote2_port", 22))
        self.remote2_process_pattern = rospy.get_param("~remote2_process_pattern", "cdpr_86.py")
        self.remote2_start_command = rospy.get_param("~remote2_start_command", "rosrun cdpr_86_actuator cdpr_86.py")
        self.remote2_setup_script = rospy.get_param("~remote2_setup_script", "~/CDPR_86/devel/setup.bash")
        self.remote2_log_file = rospy.get_param("~remote2_log_file", "~/remote2_bootstrap.log")
        self.remote2_enabled = _as_bool(rospy.get_param("~remote2_enabled", True))
        self.stop_remote2_on_shutdown = _as_bool(rospy.get_param("~stop_remote2_on_shutdown", True))
        self.remote2_imu_topic = rospy.get_param("~remote2_imu_topic", "/imu")
        self.remote2_imu_wait_timeout = float(rospy.get_param("~remote2_imu_wait_timeout", 20.0))

        self.local_ekf_command = rospy.get_param(
            "~local_ekf_command",
            "rosrun cdpr_86_host cdpr_euler_ekf_ros_node.py",
        )
        self.stop_remote_on_shutdown = _as_bool(rospy.get_param("~stop_remote_on_shutdown", True))

        self.ekf_process = None
        self.remote_ssh_process = None
        self.remote2_ssh_process = None
        self._cleanup_done = False

        rospy.loginfo(
            "Bootstrap params: remote=%s@%s:%d, setup=%s, pattern=%s, start_cmd=%s, motor_topic=%s",
            self.remote_user,
            self.remote_host,
            self.remote_port,
            self.remote_setup_script,
            self.remote_process_pattern,
            self.remote_start_command,
            self.motor_topic,
        )
        rospy.loginfo(
            "Bootstrap remote2 params: enabled=%s, remote=%s@%s:%d, setup=%s, pattern=%s, start_cmd=%s",
            str(self.remote2_enabled),
            self.remote2_user,
            self.remote2_host,
            self.remote2_port,
            self.remote2_setup_script,
            self.remote2_process_pattern,
            self.remote2_start_command,
        )
        rospy.loginfo(
            "Bootstrap remote2 readiness: imu_topic=%s, timeout=%.1fs",
            self.remote2_imu_topic,
            self.remote2_imu_wait_timeout,
        )

    @staticmethod
    def _expand_remote_home(path: str) -> str:
        if path.startswith("~/"):
            return "$HOME/" + path[2:]
        return path

    def _ssh_target(self) -> str:
        return f"{self.remote_user}@{self.remote_host}" if self.remote_user else self.remote_host

    def _ssh_target2(self) -> str:
        return f"{self.remote2_user}@{self.remote2_host}" if self.remote2_user else self.remote2_host

    def _run_ssh(self, remote_bash_command: str, *, use_remote2: bool = False) -> subprocess.CompletedProcess:
        remote_cmd = f"bash -lc {shlex.quote(remote_bash_command)}"
        port = self.remote2_port if use_remote2 else self.remote_port
        target = self._ssh_target2() if use_remote2 else self._ssh_target()
        cmd = ["ssh", "-p", str(port), target, remote_cmd]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _restart_remote_motor_node(self) -> None:
        setup_script = self._expand_remote_home(self.remote_setup_script)
        wrapped_start = (
            f'if [ -f "{setup_script}" ]; then source "{setup_script}"; fi; '
            f"exec {self.remote_start_command}"
        )

        kill_script = (
            "set +e; "
            f"pids=$(pgrep -f {shlex.quote(self.remote_process_pattern)} || true); "
            "target_pids=''; "
            "for p in $pids; do "
            "if [ \"$p\" = \"$$\" ]; then continue; fi; "
            "if kill -0 \"$p\" 2>/dev/null; then target_pids=\"$target_pids $p\"; fi; "
            "done; "
            "if [ -n \"$target_pids\" ]; then "
            "echo '[bootstrap] remote motor process found, restarting'; "
            "kill $target_pids 2>/dev/null || true; "
            "sleep 0.5; "
            "else "
            "echo '[bootstrap] remote motor process not running, starting'; "
            "fi"
        )
        kill_result = self._run_ssh(kill_script)
        if kill_result.stdout.strip():
            rospy.loginfo(kill_result.stdout.strip())
        if kill_result.stderr.strip():
            rospy.logwarn(kill_result.stderr.strip())

        ssh_cmd = [
            "ssh",
            "-p",
            str(self.remote_port),
            self._ssh_target(),
            f"bash -lc {shlex.quote(wrapped_start)}",
        ]
        rospy.loginfo("Starting remote node via SSH foreground process.")
        self.remote_ssh_process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )

        time.sleep(1.0)
        if self.remote_ssh_process.poll() is not None:
            out, err = self.remote_ssh_process.communicate(timeout=1.0)
            raise RuntimeError(
                "Remote start command exited immediately.\n"
                f"ssh stdout:\n{out or ''}\n"
                f"ssh stderr:\n{err or ''}"
            )

    def _restart_remote2_node(self) -> None:
        if not self.remote2_enabled:
            rospy.loginfo("remote2_enabled=false; skip remote2 restart.")
            return
        if not self.remote2_host or not self.remote2_start_command:
            raise RuntimeError("remote2_host/remote2_start_command is empty.")

        setup_script = self._expand_remote_home(self.remote2_setup_script) if self.remote2_setup_script else ""
        wrapped_start = (
            (f'if [ -f "{setup_script}" ]; then source "{setup_script}"; fi; ' if setup_script else "")
            + f"exec {self.remote2_start_command}"
        )

        if self.remote2_process_pattern:
            kill_script = (
                "set +e; "
                f"pids=$(pgrep -f {shlex.quote(self.remote2_process_pattern)} || true); "
                "target_pids=''; "
                "for p in $pids; do "
                "if [ \"$p\" = \"$$\" ]; then continue; fi; "
                "if kill -0 \"$p\" 2>/dev/null; then target_pids=\"$target_pids $p\"; fi; "
                "done; "
                "if [ -n \"$target_pids\" ]; then "
                "echo '[bootstrap] remote2 process found, restarting'; "
                "kill $target_pids 2>/dev/null || true; "
                "sleep 0.5; "
                "else "
                "echo '[bootstrap] remote2 process not running, starting'; "
                "fi"
            )
            kill_result = self._run_ssh(kill_script, use_remote2=True)
            if kill_result.stdout.strip():
                rospy.loginfo(kill_result.stdout.strip())
            if kill_result.stderr.strip():
                rospy.logwarn(kill_result.stderr.strip())

        ssh_cmd = [
            "ssh",
            "-p",
            str(self.remote2_port),
            self._ssh_target2(),
            f"bash -lc {shlex.quote(wrapped_start)}",
        ]
        rospy.loginfo("Starting remote2 node via SSH foreground process.")
        self.remote2_ssh_process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        time.sleep(1.0)
        if self.remote2_ssh_process.poll() is not None:
            out, err = self.remote2_ssh_process.communicate(timeout=1.0)
            raise RuntimeError(
                "Remote2 start command exited immediately.\n"
                f"ssh stdout:\n{out or ''}\n"
                f"ssh stderr:\n{err or ''}"
            )

    def _wait_motor_topic_ready(self) -> None:
        try:
            rospy.wait_for_message(
                self.motor_topic,
                Float32MultiArray,
                timeout=self.motor_topic_wait_timeout,
            )
        except rospy.ROSException as exc:
            raise RuntimeError(
                f"Timeout waiting for {self.motor_topic} (>{self.motor_topic_wait_timeout}s)."
            ) from exc

    def _wait_remote2_imu_ready(self) -> None:
        if not self.remote2_enabled:
            rospy.loginfo("remote2_enabled=false; skip IMU readiness wait.")
            return
        try:
            rospy.wait_for_message(
                self.remote2_imu_topic,
                Imu,
                timeout=self.remote2_imu_wait_timeout,
            )
        except rospy.ROSException as exc:
            raise RuntimeError(
                f"Timeout waiting for {self.remote2_imu_topic} (>{self.remote2_imu_wait_timeout}s)."
            ) from exc

    def _tail_remote_log(self, lines: int = 80) -> str:
        remote_log_file = self._expand_remote_home(self.remote_log_file)
        cmd = (
            "set +e; "
            f'if [ -f "{remote_log_file}" ]; then '
            f'tail -n {int(lines)} "{remote_log_file}"; '
            "else "
            f"echo '[bootstrap] remote log file not found: {remote_log_file}'; "
            "fi"
        )
        result = self._run_ssh(cmd)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        return f"{out}\n[stderr]\n{err}".strip() if err else out

    def _tail_remote2_log(self, lines: int = 80) -> str:
        remote_log_file = self._expand_remote_home(self.remote2_log_file)
        cmd = (
            "set +e; "
            f'if [ -f "{remote_log_file}" ]; then '
            f'tail -n {int(lines)} "{remote_log_file}"; '
            "else "
            f"echo '[bootstrap] remote2 log file not found: {remote_log_file}'; "
            "fi"
        )
        result = self._run_ssh(cmd, use_remote2=True)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        return f"{out}\n[stderr]\n{err}".strip() if err else out

    def _start_local_ekf(self) -> None:
        rospy.loginfo("Starting local EKF: %s", self.local_ekf_command)
        self.ekf_process = subprocess.Popen(["bash", "-lc", self.local_ekf_command], preexec_fn=os.setsid)

    def _stop_local_ekf(self) -> None:
        if self.ekf_process is None or self.ekf_process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.ekf_process.pid), signal.SIGTERM)
        except Exception:
            pass

    def _stop_remote_motor_node(self) -> None:
        if self.remote_ssh_process is not None and self.remote_ssh_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.remote_ssh_process.pid), signal.SIGTERM)
            except Exception:
                pass

        setup_script = self._expand_remote_home(self.remote_setup_script)
        stop_script = (
            "set +e; "
            f'if [ -f "{setup_script}" ]; then source "{setup_script}"; fi; '
            f"pids=$(pgrep -f {shlex.quote(self.remote_process_pattern)} || true); "
            "target_pids=''; "
            "for p in $pids; do "
            "if [ \"$p\" = \"$$\" ]; then continue; fi; "
            "if kill -0 \"$p\" 2>/dev/null; then target_pids=\"$target_pids $p\"; fi; "
            "done; "
            "if [ -n \"$target_pids\" ]; then "
            f"echo '[bootstrap] stopping remote motor process by pattern: {self.remote_process_pattern}'; "
            "kill $target_pids 2>/dev/null || true; "
            "else "
            "echo '[bootstrap] no remote motor process to stop'; "
            "fi"
        )
        result = self._run_ssh(stop_script)
        if result.stdout.strip():
            rospy.loginfo(result.stdout.strip())
        if result.stderr.strip():
            rospy.logwarn(result.stderr.strip())

    def _stop_remote2_node(self) -> None:
        if not self.remote2_enabled:
            return
        if self.remote2_ssh_process is not None and self.remote2_ssh_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.remote2_ssh_process.pid), signal.SIGTERM)
            except Exception:
                pass

        if not self.remote2_process_pattern or not self.stop_remote2_on_shutdown:
            return

        setup_script = self._expand_remote_home(self.remote2_setup_script) if self.remote2_setup_script else ""
        stop_script = (
            "set +e; "
            + (f'if [ -f "{setup_script}" ]; then source "{setup_script}"; fi; ' if setup_script else "")
            + f"pids=$(pgrep -f {shlex.quote(self.remote2_process_pattern)} || true); "
            "target_pids=''; "
            "for p in $pids; do "
            "if [ \"$p\" = \"$$\" ]; then continue; fi; "
            "if kill -0 \"$p\" 2>/dev/null; then target_pids=\"$target_pids $p\"; fi; "
            "done; "
            "if [ -n \"$target_pids\" ]; then "
            f"echo '[bootstrap] stopping remote2 process by pattern: {self.remote2_process_pattern}'; "
            "kill $target_pids 2>/dev/null || true; "
            "else "
            "echo '[bootstrap] no remote2 process to stop'; "
            "fi"
        )
        result = self._run_ssh(stop_script, use_remote2=True)
        if result.stdout.strip():
            rospy.loginfo(result.stdout.strip())
        if result.stderr.strip():
            rospy.logwarn(result.stderr.strip())

    def cleanup(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        self._stop_local_ekf()
        if self.stop_remote_on_shutdown:
            self._stop_remote_motor_node()
        self._stop_remote2_node()

    def run(self) -> int:
        rospy.loginfo("Restart remote motor node and then launch local EKF.")
        self._restart_remote_motor_node()
        if self.remote2_enabled:
            self._restart_remote2_node()
        time.sleep(self.post_restart_wait_s)
        self._wait_motor_topic_ready()
        if self.remote2_enabled:
            self._wait_remote2_imu_ready()
        rospy.loginfo("%s is ready.", self.motor_topic)
        self._start_local_ekf()
        return self.ekf_process.wait()


def main() -> None:
    rospy.init_node("remote_motor_ekf_bootstrap", anonymous=False)
    node = RemoteMotorEkfBootstrap()
    rospy.on_shutdown(node.cleanup)
    try:
        code = node.run()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by Ctrl+C.")
        code = 130
    except Exception as exc:
        rospy.logerr(str(exc))
        tail = node._tail_remote_log()
        if tail:
            rospy.logerr("Remote log tail:\n%s", tail)
        if node.remote2_enabled:
            tail2 = node._tail_remote2_log()
            if tail2:
                rospy.logerr("Remote2 log tail:\n%s", tail2)
        code = 1
    node.cleanup()
    sys.exit(code)


if __name__ == "__main__":
    main()
