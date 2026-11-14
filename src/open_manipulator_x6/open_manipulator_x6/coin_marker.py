#!/usr/bin/env python3
"""Coin count marker for the pad calibration: publishes the load on the pad as std_msgs/Float32
on /calib_load_g, so it is recorded in the same bag as the pad frames.

    python3 coin_marker.py                 # plate weight asked at start
    python3 coin_marker.py --plate 3.2     # plate weight in grams
    python3 coin_marker.py --coin 4.5      # 10-yen coins (1-yen 1.00, 5-yen 3.75, 10-yen 4.50,
                                           # 50-yen 4.00, 100-yen 4.80)

Keys (then Enter):  Enter = +1 coin   - = -1 coin   p = plate on/off   0 = all off   q = quit
The current load is re-published 10 times a second, so every moment in the bag has a label.
"""
import argparse
import threading

import rclpy
from std_msgs.msg import Float32



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", type=float, default=4.5, help="grams per coin (10-yen = 4.5)")
    ap.add_argument("--plate", type=float, default=None, help="plate weight in grams")
    a = ap.parse_args()
    plate_g = a.plate if a.plate is not None else float(input("Plate weight (g): ") or 0)

    rclpy.init()
    node = rclpy.create_node("coin_marker")
    pub = node.create_publisher(Float32, "/calib_load_g", 10)
    state = {"coins": 0, "plate": False}

    def load():
        return (plate_g if state["plate"] else 0.0) + state["coins"] * a.coin

    node.create_timer(0.1, lambda: pub.publish(Float32(data=float(load()))))
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("Enter = +1 coin, '-' = -1, 'p' = plate on/off, '0' = all off, 'q' = quit")
    while True:
        k = input(f"[plate {'on ' if state['plate'] else 'off'} | coins {state['coins']:2d} | "
                  f"load {load():5.2f} g] > ").strip().lower()
        if k == "q":
            break
        if k == "":
            state["coins"] += 1
        elif k == "-":
            state["coins"] = max(0, state["coins"] - 1)
        elif k == "p":
            state["plate"] = not state["plate"]
        elif k == "0":
            state["coins"], state["plate"] = 0, False
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()