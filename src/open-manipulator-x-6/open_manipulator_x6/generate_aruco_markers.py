#!/usr/bin/env python3
"""
Make printable ArUco markers for the X6 workspace.

    python3 generate_aruco_markers.py                       # -> ./aruco_markers/
    python3 generate_aruco_markers.py --table-mm 40 --bottle-mm 30 --out ~/aruco

Writes:
    aruco_markers.pdf      A4 sheet, true scale (print at 100% / "actual size", not "fit to page")
    marker_<id>_<name>.png one image per marker (for checking, not for printing)

Markers (DICT_4X4_50):
    0 base          on the bar beside the arm base; point its arrow to the robot's front (+x)
    1 bottle_start  beside the bottle's start spot
    2 bottle_place  beside the bottle's place spot
    3 bottle        on the bottle's side, below the cap

The size printed under each marker is the outer edge of its black square: that is the number
the detector needs. After printing, measure the 50 mm scale bar; if it is not 50 mm, your printer
scaled the page.
"""
import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

A4_W, A4_H = 210.0, 297.0


def get_dictionary(name):
    const = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(const)
    return cv2.aruco.Dictionary_get(const)


def marker_bits(dictionary, marker_id):
    n = dictionary.markerSize + 2                      # one black border cell on each side
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, n, borderBits=1)
    return cv2.aruco.drawMarker(dictionary, marker_id, n, borderBits=1)


def draw_marker(ax, bits, x0, y0, size_mm):
    """Draw one marker as vector squares; (x0, y0) = bottom-left corner in mm."""
    n = bits.shape[0]
    cell = size_mm / n
    for r in range(n):
        for c in range(n):
            if bits[r, c] < 128:
                ax.add_patch(Rectangle((x0 + c * cell, y0 + (n - 1 - r) * cell), cell, cell,
                                       facecolor="black", edgecolor="black", linewidth=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--table-mm", type=float, default=40.0, help="size of markers 0-2")
    ap.add_argument("--bottle-mm", type=float, default=30.0, help="size of marker 3")
    ap.add_argument("--out", default="aruco_markers")
    args = ap.parse_args()

    markers = [(0, "base", args.table_mm), (1, "bottle_start", args.table_mm),
               (2, "bottle_place", args.table_mm), (3, "bottle", args.bottle_mm)]
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    d = get_dictionary(args.dict)

    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, A4_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(A4_W / 2, A4_H - 15, f"X6 workspace markers ({args.dict})", ha="center", fontsize=13)
    ax.text(A4_W / 2, A4_H - 22, "Print at 100% (actual size). Cut along the grey lines, "
            "keeping the white border.", ha="center", fontsize=8, color="0.3")

    cols = [(30.0, 175.0), (120.0, 175.0), (30.0, 80.0), (120.0, 80.0)]   # bottom-left of each cell
    for (mid, name, size), (cx, cy) in zip(markers, cols):
        bits = marker_bits(d, mid)
        cv2.imwrite(os.path.join(out, f"marker_{mid}_{name}.png"),
                    cv2.resize(bits, (bits.shape[1] * 100, bits.shape[0] * 100),
                               interpolation=cv2.INTER_NEAREST))
        x0, y0 = cx + (60 - size) / 2, cy + (60 - size) / 2
        draw_marker(ax, bits, x0, y0, size)
        m = max(8.0, size / 6 * 1.5)                    # white margin, at least 1.5 cells
        top = 16.0 if name == "base" else 0.0           # extra room above the base marker for its arrow
        ax.add_patch(Rectangle((x0 - m, y0 - m), size + 2 * m, size + 2 * m + top, fill=False,
                               edgecolor="0.6", linewidth=0.5, linestyle=(0, (3, 2))))
        ax.text(x0 + size / 2, y0 - m - 5, f"ID {mid}: {name}  ({size:.0f} mm)",
                ha="center", fontsize=9)
        if name == "base":
            # Arrow = the marker's +y direction, kept clear of the white quiet zone but inside the cut line.
            ax.add_patch(FancyArrow(x0 + size / 2, y0 + size + m, 0, 9, width=1.0,
                                    head_width=4, head_length=4, color="0.2", length_includes_head=True))
            ax.text(x0 + size / 2 + 4, y0 + size + m + 4, "robot front (+x)", fontsize=7, va="center")

    # scale bar
    ax.add_patch(Rectangle((30, 30), 50, 3, color="black"))
    for x in (30, 80):
        ax.plot([x, x], [27, 36], color="black", linewidth=0.8)
    ax.text(55, 38, "50 mm - measure after printing", ha="center", fontsize=8)

    pdf = os.path.join(out, "aruco_markers.pdf")
    fig.savefig(pdf)
    print(f"Wrote {pdf} and {len(markers)} PNGs in {out}")


if __name__ == "__main__":
    main()