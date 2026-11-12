#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture Kinect 360 depth + colour as 3D .ply files (point cloud or mesh).

  python3 kinect3d.py snap                 # one scan -> scan_<time>.ply
  python3 kinect3d.py snap --mesh          # same, but as a triangulated surface
  python3 kinect3d.py record --fps 5       # sequence -> rec_<time>/frame_00001.ply ... (Ctrl+C stops)

Coordinates are in metres, Blender convention: X right, Y away from the camera, Z up.
"""
import argparse
import ctypes
import json
import os
import signal
import sys
import time
import warnings

import numpy as np

W, H = 640, 480
FREENECT_VIDEO_RGB = 0
FREENECT_DEPTH_REGISTERED = 4  # depth in mm, aligned to the RGB camera, 0 = no reading

# Intrinsics of the Kinect 360 RGB camera (registered depth lives in this frame).
FX, FY, CX, CY = 529.2, 525.6, 328.9, 267.5

lib = ctypes.CDLL("libfreenect_sync.so.0")
lib.freenect_sync_get_depth.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int]
lib.freenect_sync_get_video.argtypes = lib.freenect_sync_get_depth.argtypes
lib.freenect_sync_set_tilt_degs.argtypes = [ctypes.c_int, ctypes.c_int]


def grab():
    """Return (depth_mm uint16 HxW, rgb uint8 HxWx3)."""
    ptr, ts = ctypes.c_void_p(), ctypes.c_uint32()
    if lib.freenect_sync_get_depth(ctypes.byref(ptr), ctypes.byref(ts), 0, FREENECT_DEPTH_REGISTERED):
        sys.exit("Could not read depth - is the Kinect plugged in and not used by another program?")
    depth = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16)), (H, W)).copy()
    if lib.freenect_sync_get_video(ctypes.byref(ptr), ctypes.byref(ts), 0, FREENECT_VIDEO_RGB):
        sys.exit("Could not read RGB video")
    rgb = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (H, W, 3)).copy()
    return depth, rgb


def to_ply(path, depth, rgb, mesh=False, max_dist=4.0, step=1, edge=0.04):
    d = depth[::step, ::step].astype(np.float32) / 1000.0
    c = rgb[::step, ::step]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u, v = u * step, v * step
    valid = (d > 0) & (d < max_dist)

    x = (u - CX) * d / FX
    z = -(v - CY) * d / FY
    idx = np.full((h, w), -1, np.int32)
    idx[valid] = np.arange(valid.sum(), dtype=np.int32)

    verts = np.empty(valid.sum(), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts["x"], verts["y"], verts["z"] = x[valid], d[valid], z[valid]
    verts["red"], verts["green"], verts["blue"] = c[valid].T

    faces = np.empty(0, [("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    if mesh:
        # Connect neighbouring pixels into triangles, skipping jumps in depth (object edges).
        tl, tr, bl, br = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        dtl, dtr, dbl, dbr = d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]
        lim = edge * step * np.maximum(dtl, 0.5)  # allow bigger gaps further away
        tris = []
        for a, b, cc, da, db, dc in ((tl, bl, tr, dtl, dbl, dtr), (tr, bl, br, dtr, dbl, dbr)):
            ok = (a >= 0) & (b >= 0) & (cc >= 0)
            ok &= (np.abs(da - db) < lim) & (np.abs(da - dc) < lim) & (np.abs(db - dc) < lim)
            tris.append(np.stack([a[ok], b[ok], cc[ok]], 1))
        tris = np.concatenate(tris)
        faces = np.empty(len(tris), faces.dtype)
        faces["n"] = 3
        faces["a"], faces["b"], faces["c"] = tris.T

    with open(path, "wb") as f:
        f.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            + (f"element face {len(faces)}\nproperty list uchar int vertex_indices\n" if mesh else "")
            + "end_header\n").encode())
        f.write(verts.tobytes())
        if mesh:
            f.write(faces.tobytes())
    return len(verts), len(faces)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["snap", "record"])
    p.add_argument("--mesh", action="store_true", help="write a triangulated surface instead of just points")
    p.add_argument("--fps", type=float, default=5, help="frames per second for record (default 5)")
    p.add_argument("--max-dist", type=float, default=4.0, help="ignore anything further than this, metres")
    p.add_argument("--step", type=int, default=1, help="use every Nth pixel (2 = 4x smaller files)")
    p.add_argument("--avg", type=int, default=5, help="snap: average N frames to reduce noise (default 5)")
    p.add_argument("--tilt", type=int, help="set the motor tilt in degrees (-27..27) before capturing")
    p.add_argument("-o", "--out", help="output file (snap) or folder (record)")
    a = p.parse_args()

    if a.tilt is not None:
        lib.freenect_sync_set_tilt_degs(a.tilt, 0)
        time.sleep(1.5)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        grab()  # warm-up: the first frames after start are often incomplete
        time.sleep(0.5)
        if a.mode == "snap":
            depths, rgb = [], None
            for _ in range(max(1, a.avg)):
                dep, rgb = grab()
                depths.append(dep)
            stack = np.stack(depths).astype(np.float32)
            stack[stack == 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                depth = np.nan_to_num(np.nanmedian(stack, 0)).astype(np.uint16)
            out = a.out or f"scan_{stamp}.ply"
            n, nf = to_ply(out, depth, rgb, a.mesh, a.max_dist, a.step)
            print(f"Saved {out}: {n:,} points" + (f", {nf:,} triangles" if a.mesh else ""))
        else:
            out = a.out or f"rec_{stamp}"
            os.makedirs(out, exist_ok=True)
            stop = []
            signal.signal(signal.SIGINT, lambda *_: stop.append(1))
            print(f"Recording to {out}/ at {a.fps} fps - press Ctrl+C to stop")
            i, t0 = 0, time.time()
            while not stop:
                t = time.time()
                depth, rgb = grab()
                i += 1
                to_ply(os.path.join(out, f"frame_{i:05d}.ply"), depth, rgb, a.mesh, a.max_dist, a.step)
                print(f"\rframe {i}  ({time.time() - t0:.0f}s)", end="", flush=True)
                time.sleep(max(0, 1 / a.fps - (time.time() - t)))
            elapsed = time.time() - t0
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump({"frames": i, "fps_target": a.fps, "fps_actual": round(i / elapsed, 2) if elapsed else a.fps,
                           "step": a.step, "mesh": a.mesh, "max_dist": a.max_dist}, f, indent=2)
            print(f"\nSaved {i} frames in {out}/ ({i / elapsed:.1f} fps actual)")
    finally:
        lib.freenect_sync_stop()


if __name__ == "__main__":
    main()
