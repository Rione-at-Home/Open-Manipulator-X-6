#!/usr/bin/env python3
"""
Estimate each link's mass, centre of mass and inertia from its STL mesh, part by part:
servo bodies get the datasheet mass, printed parts get PLA at an effective density.

    python3 tools/compute_inertials.py meshes/            # prints the <inertial> blocks

The meshes are the SolidWorks export (link frame, metres). Each link mesh is split into its
connected parts; the servo body is recognised by size (about 47 x 41 x 29 mm, 42-43 cm3; an
XL430 mesh comes in several pieces, all inside that box). Edit SERVOS / PLA_DENSITY below, or
set measured link masses in MEASURED_G to scale the printed parts so the total matches.
"""
import sys

import numpy as np
import trimesh

PLA_DENSITY = 1240.0 * 0.75        # kg/m3: solid PLA x ~0.75 for 60 % infill with solid walls
SERVO_KG = {"XM430": 0.082, "XL430": 0.0572}
SERVOS = {                          # which servo body sits in which link
    "base_link": "XM430", "link_1": "XM430", "link_2": "XM430", "link_3": "XM430",
    "link_4": "XL430", "link_5": "XL430", "gripper_base_link": "XM430",
}
EXTRA_KG = {"left_finger_link": 0.003, "right_finger_link": 0.003}   # FlexiTac pad + flex cable
MEASURED_G = {}                     # e.g. {"link_2": 112.0} once weighed (whole link incl. servo)
LINKS = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
         "gripper_base_link", "left_finger_link", "right_finger_link"]
SERVO_BOX_MM = np.array([48.0, 42.5, 30.0])   # bounding box that contains a servo body (sorted)


def is_servo_piece(p, servo_box_min, servo_box_max):
    lo, hi = p.bounds
    return np.all(lo >= servo_box_min - 1e-4) and np.all(hi <= servo_box_max + 1e-4)


def servo_region(parts):
    """Box that holds the servo body. An XM430 mesh is one closed piece (~43 cm3); an XL430 mesh
    is a middle body (~30 cm3, 23 mm thick) plus two case caps (~6 mm and ~5 mm) on its thin
    sides, so for a split body the box is grown by 12 mm both ways along its thinnest axis."""
    for p in sorted(parts, key=lambda q: -abs(q.volume)):
        ext = (p.bounds[1] - p.bounds[0]) * 1000
        if np.all(np.sort(ext)[::-1] <= SERVO_BOX_MM) and ext.max() > 40:
            lo, hi = p.bounds.copy()
            if abs(p.volume) * 1e6 < 38.0:            # split body: add the caps' room
                k = int(np.argmin(ext))
                lo[k] -= 0.012
                hi[k] += 0.012
            return lo - 1e-3, hi + 1e-3
    return None


def group_props(meshes, mass):
    """Uniform solid made of these mesh pieces (signed volumes, so hole surfaces subtract),
    scaled to `mass`. Returns (mass, com, inertia about com)."""
    g = trimesh.util.concatenate(meshes)
    vol = g.volume
    if vol <= 0:
        raise ValueError("group has no enclosed volume")
    g.density = mass / vol
    return mass, np.asarray(g.center_mass), np.asarray(g.moment_inertia)


def combine(groups):
    total = sum(m for m, _, _ in groups)
    com = sum(m * c for m, c, _ in groups) / total
    I = np.zeros((3, 3))
    for m, c, Ig in groups:
        d = c - com
        I += Ig + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total, com, I


def link_inertial(path, link, density=None):
    """(mass, com, inertia, has_servo) for one link."""
    pieces = [p for p in trimesh.load(path).split(only_watertight=False) if len(p.faces) > 2]
    region = servo_region([p for p in pieces if abs(p.volume) > 5e-8]) if link in SERVOS else None
    servo = [p for p in pieces if region is not None and is_servo_piece(p, *region)]
    printed = [p for p in pieces if not any(p is s for s in servo)]
    rho = PLA_DENSITY if density is None else density
    groups = []
    if servo:
        groups.append(group_props(servo, SERVO_KG[SERVOS[link]]))
    printed_vol = trimesh.util.concatenate(printed).volume if printed else 0.0
    extra = EXTRA_KG.get(link, 0.0)
    if link in MEASURED_G and density is None and printed_vol > 0:
        servo_kg = SERVO_KG[SERVOS[link]] if servo else 0.0
        return link_inertial(path, link, (MEASURED_G[link] / 1000.0 - servo_kg - extra) / printed_vol)
    if printed_vol > 0:
        groups.append(group_props(printed, rho * printed_vol + extra))
    m, com, I = combine(groups)
    return m, com, I, bool(servo)


def cad_check(path):
    """Mass properties at density 1000 kg/m3, for comparison with the SolidWorks export."""
    pieces = [p for p in trimesh.load(path).split(only_watertight=False) if len(p.faces) > 2]
    vol = trimesh.util.concatenate(pieces).volume
    return group_props(pieces, 1000.0 * vol)


def inertial_xml(m, com, I):
    return (f'<inertial>\n  <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0" />\n'
            f'  <mass value="{m:.5f}" />\n'
            f'  <inertia ixx="{I[0,0]:.4e}" ixy="{I[0,1]:.4e}" ixz="{I[0,2]:.4e}" '
            f'iyy="{I[1,1]:.4e}" iyz="{I[1,2]:.4e}" izz="{I[2,2]:.4e}" />\n</inertial>')


if __name__ == "__main__":
    import os
    d = sys.argv[1] if len(sys.argv) > 1 else "meshes"
    total = 0.0
    for link in LINKS:
        m, com, I, servo = link_inertial(os.path.join(d, f"{link}.STL"), link)
        total += m
        print(f"<!-- {link}: {m * 1000:.1f} g ({'servo ' + SERVOS[link] if servo else 'printed only'}) -->")
        print(inertial_xml(m, com, I))
    print(f"<!-- total {total * 1000:.0f} g -->")
