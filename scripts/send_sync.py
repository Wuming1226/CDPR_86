#!/usr/bin/env python3

import argparse
import time

import can


def parse_args():
    parser = argparse.ArgumentParser(description="Send CANopen SYNC frames at fixed rate.")
    parser.add_argument("--channel", default="can0", help="SocketCAN channel name (default: can0)")
    parser.add_argument("--bustype", default="socketcan", help="python-can bus type (default: socketcan)")
    parser.add_argument("--hz", type=float, default=30.0, help="SYNC frequency in Hz (default: 30.0)")
    parser.add_argument("--sync-id", type=lambda x: int(x, 0), default=0x080, help="SYNC COB-ID (default: 0x080)")
    parser.add_argument(
        "--data",
        default="01",
        help="SYNC data bytes as hex string, e.g. '01' or '01,02' (default: 01)",
    )
    return parser.parse_args()


def parse_data_bytes(data_str):
    if not data_str.strip():
        return []
    parts = data_str.replace(" ", "").split(",")
    return [int(p, 16) for p in parts if p]


def main():
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("--hz must be > 0")

    period = 1.0 / args.hz
    payload = parse_data_bytes(args.data)

    bus = can.interface.Bus(channel=args.channel, bustype=args.bustype)
    sync_msg = can.Message(
        arbitration_id=args.sync_id,
        data=payload,
        is_extended_id=False,
    )

    sent_count = 0
    next_time = time.monotonic()
    print(
        "Sending SYNC on {} at {:.3f} Hz, id=0x{:03X}, data={}".format(
            args.channel, args.hz, args.sync_id, payload
        )
    )

    try:
        while True:
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.001, next_time - now))
                continue

            bus.send(sync_msg)
            sent_count += 1
            next_time += period

            if sent_count % max(1, int(args.hz)) == 0:
                print("sent {} SYNC frames".format(sent_count))
    except KeyboardInterrupt:
        print("\nStopped. Total SYNC frames sent: {}".format(sent_count))


if __name__ == "__main__":
    main()

