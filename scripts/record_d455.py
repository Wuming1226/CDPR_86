#!/usr/bin/env python3
"""
Record Intel RealSense D455 video to MP4.

Notes:
- Requires RealSense SDK + pyrealsense2 python package.
- This script records the color stream (BGR) by default.
- Optionally, it also records depth as a colored visualization side-by-side.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RECORDINGS_ROOT = _SCRIPT_DIR.parent / "recordings"


def _import_deps():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `pyrealsense2`. Install it with:\n"
            "  pip install pyrealsense2\n"
            "and ensure Intel RealSense SDK is installed."
        ) from e

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `opencv-python`. Install it with:\n"
            "  pip install opencv-python"
        ) from e

    import numpy as np  # type: ignore

    return rs, cv2, np


def _list_devices(rs) -> list[str]:
    serials: list[str] = []
    for dev in rs.context().query_devices():
        try:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return serials


def _select_device(rs, serial: str | None):
    devices = rs.context().query_devices()
    if devices.size() == 0:
        raise RuntimeError("No RealSense device detected. Plug in D455 and retry.")

    if serial is None:
        return devices[0]

    for dev in devices:
        try:
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return dev
        except Exception:
            continue

    available = _list_devices(rs)
    raise RuntimeError(f"RealSense serial={serial} not found. Available: {available}")


def _pick_max_color_profile(rs, device, fps: int) -> tuple[int, int, int]:
    """
    Pick the max-resolution color profile at the given fps.
    Returns (width, height, fps_selected).
    """
    # Find a color sensor.
    color_sensor = None
    for s in device.sensors:
        try:
            # Heuristic: color sensor usually exposes stream.color profiles
            profiles = s.get_stream_profiles()
            has_color = any(p.stream_type() == rs.stream.color for p in profiles)
            if has_color:
                color_sensor = s
                break
        except Exception:
            continue

    if color_sensor is None:
        raise RuntimeError("No color sensor found on this RealSense device.")

    candidates: list[tuple[int, int, int]] = []
    for p in color_sensor.get_stream_profiles():
        try:
            if p.stream_type() != rs.stream.color:
                continue
            if p.is_video_stream_profile() is False:
                continue
            vp = p.as_video_stream_profile()
            w, h = int(vp.width()), int(vp.height())
            pfps = int(vp.fps())
            fmt = vp.format()
            if pfps != int(fps):
                continue
            # Prefer RGB8 for compatibility, but accept others.
            if fmt not in (rs.format.rgb8, rs.format.bgr8, rs.format.yuyv, rs.format.mjpeg):
                continue
            candidates.append((w, h, pfps))
        except Exception:
            continue

    if not candidates:
        # Fallback: ignore fps and pick the max area profile.
        for p in color_sensor.get_stream_profiles():
            try:
                if p.stream_type() != rs.stream.color or not p.is_video_stream_profile():
                    continue
                vp = p.as_video_stream_profile()
                w, h = int(vp.width()), int(vp.height())
                pfps = int(vp.fps())
                fmt = vp.format()
                if fmt not in (rs.format.rgb8, rs.format.bgr8, rs.format.yuyv, rs.format.mjpeg):
                    continue
                candidates.append((w, h, pfps))
            except Exception:
                continue

    if not candidates:
        raise RuntimeError("No usable color stream profiles found.")

    # Sort by area desc, then by fps desc.
    candidates.sort(key=lambda t: (t[0] * t[1], t[2]), reverse=True)
    return candidates[0]


def _timestamp_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="Record Intel RealSense D455 to MP4.")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial number to use.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_RECORDINGS_ROOT,
        help=f"Output directory (default: {_DEFAULT_RECORDINGS_ROOT}).",
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="FPS for recording.")
    parser.add_argument(
        "--max-color",
        action="store_true",
        help="Auto-pick the maximum supported COLOR resolution (at --fps when possible).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds. 0 means until Ctrl+C.",
    )
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help="Also output a depth visualization (jet colormap) side-by-side.",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.2,
        help="Depth visualization min range (meters).",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=4.0,
        help="Depth visualization max range (meters).",
    )
    parser.add_argument(
        "--depth-jet",
        action="store_true",
        help="Use JET colormap for depth (default if --with-depth).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Record only depth visualization (requires --with-depth).",
    )
    args = parser.parse_args()

    rs, cv2, np = _import_deps()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output_dir={output_dir}")

    # Device selection / sanity check.
    serials = _list_devices(rs)
    if args.serial is None and len(serials) > 1:
        print(
            f"[INFO] multiple RealSense devices detected: {serials}. Pass --serial to select one.",
            file=sys.stderr,
        )

    device = _select_device(rs, args.serial)
    if args.max_color and not args.with_depth and not args.no_color:
        w, h, fps_sel = _pick_max_color_profile(rs, device, int(args.fps))
        args.width, args.height, args.fps = int(w), int(h), int(fps_sel)

    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial is not None:
        config.enable_device(args.serial)

    # Color stream: request RGB8 and convert to BGR for OpenCV writing.
    if not args.no_color or args.with_depth:
        config.enable_stream(
            rs.stream.color,
            args.width,
            args.height,
            rs.format.rgb8,
            args.fps,
        )

    # Depth stream for visualization.
    if args.with_depth:
        config.enable_stream(
            rs.stream.depth,
            args.width,
            args.height,
            rs.format.z16,
            args.fps,
        )

    align = rs.align(rs.stream.color) if args.with_depth else None

    # Start pipeline.
    profile = pipeline.start(config)
    try:
        sensor = profile.get_device().first_depth_sensor
        depth_scale = float(sensor.get_depth_scale())
    except Exception:
        depth_scale = None

    # Prepare video writer.
    tag = _timestamp_tag()
    base_name = f"d455_{tag}_color{'' if not args.with_depth else '_depth'}.mp4"
    output_path = output_dir / base_name

    # Determine output frame shape.
    if args.with_depth:
        # Side-by-side: either (color | depth) or (depth alone if --no-color).
        # We'll render color on left, depth on right when both are enabled.
        depth_w = args.width
        depth_h = args.height
        if args.no_color:
            out_w = depth_w
        else:
            out_w = args.width * 2
        out_h = args.height
    else:
        out_w = args.width
        out_h = args.height

    # Choose codec.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(args.fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    depth_colormap = cv2.COLORMAP_JET if args.depth_jet else cv2.COLORMAP_JET

    print(
        f"[INFO] Recording to: {output_path}\n"
        f"       resolution: {args.width}x{args.height}, fps={args.fps}\n"
        f"       duration: {args.duration if args.duration > 0 else 'until Ctrl+C'}\n"
        f"       with_depth: {args.with_depth}, no_color: {args.no_color}"
    )

    start_t = time.time()
    frame_count = 0

    try:
        while True:
            if args.duration and args.duration > 0:
                if time.time() - start_t >= args.duration:
                    break

            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_img = None
            if not args.no_color:
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                # RGB8 -> BGR for OpenCV.
                rgb = np.asanyarray(color_frame.get_data())
                color_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            depth_vis = None
            if args.with_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue

                depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
                if depth_scale is None:
                    # Fallback: assume raw is already meters (unlikely but prevents crash).
                    depth_m = depth_raw.astype(np.float32)
                else:
                    depth_m = depth_raw.astype(np.float32) * depth_scale

                # Clip and normalize to 8-bit for visualization.
                depth_clipped = np.clip(depth_m, args.depth_min, args.depth_max)
                depth_norm = (depth_clipped - args.depth_min) / max(1e-6, (args.depth_max - args.depth_min))
                depth_8u = (depth_norm * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_8u, depth_colormap)

            # Compose output frame.
            if args.with_depth:
                if args.no_color:
                    out = depth_vis
                else:
                    # side-by-side (color left, depth right)
                    out = np.concatenate([color_img, depth_vis], axis=1)
            else:
                out = color_img

            writer.write(out)
            frame_count += 1
    except KeyboardInterrupt:
        pass
    finally:
        writer.release()
        pipeline.stop()

    elapsed = time.time() - start_t
    fps_eff = frame_count / elapsed if elapsed > 1e-6 else 0.0
    print(f"[INFO] Finished. frames={frame_count}, elapsed={elapsed:.2f}s, fps_eff={fps_eff:.2f}")
    print(f"[INFO] Output: {output_path}")


if __name__ == "__main__":
    main()

