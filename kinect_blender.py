"""Load a kinect3d.py recording (folder of frame_XXXXX.ply) into Blender as an animation.

From a terminal:
    blender --python kinect_blender.py -- rec_20261111_173235
    blender --python kinect_blender.py -- rec_20261111_173235 --save scene.blend
    blender --python kinect_blender.py -- rec_20261111_173235 --mesh    (join points into surfaces)
    blender -b --python kinect_blender.py -- rec_... --save scene.blend    (no window, just build the .blend)

Or inside Blender: Scripting tab -> open this file -> set FOLDER below -> Run Script.

Each frame becomes its own object that is only visible on its own frame, so pressing
Space plays the 3D sequence. A camera named "Kinect" sits where the real Kinect was.
"""
import glob
import json
import math
import os
import sys

import bpy
import numpy as np

FOLDER = ""        # used when run from inside Blender without command-line arguments
POINT_SIZE = 1.0   # dot size relative to the gap between neighbouring points (point clouds only)

MESH = False       # True = join points into a surface (same as --mesh on the command line)
EDGE = 0.04        # --mesh: don't join neighbours whose depth differs by more than ~4% (object edges)

# Kinect 360 RGB camera intrinsics (must match kinect3d.py) and image size
FX, FY, CX, CY, W, H = 529.2, 525.6, 328.9, 267.5, 640, 480


def to_pixels(xyz):
    """Project points back to the Kinect image -> (u, v) pixel coordinates and depth."""
    d = np.maximum(xyz[:, 1], 1e-6)
    return xyz[:, 0] * FX / d + CX, -xyz[:, 2] * FY / d + CY, d


def detect_step(xyz):
    """Pixel spacing the recording was made with (--step), from the projected point positions."""
    u = np.unique(np.round(to_pixels(xyz[:20000])[0]))
    return max(1, int(np.median(np.diff(u)))) if len(u) > 1 else 1


def grid_triangles(xyz, step):
    """Rebuild a surface from a point cloud: join points that were neighbouring pixels on the depth image."""
    u, v, d = to_pixels(xyz)
    gu, gv = np.round(u / step).astype(np.int64), np.round(v / step).astype(np.int64)
    h, w = H // step + 2, W // step + 2
    ok = (gu >= 0) & (gu < w) & (gv >= 0) & (gv < h)
    idx = np.full((h, w), -1, np.int32)
    dep = np.zeros((h, w), np.float32)
    idx[gv[ok], gu[ok]] = np.nonzero(ok)[0]
    dep[gv[ok], gu[ok]] = d[ok]
    tl, tr, bl, br = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    dtl, dtr, dbl, dbr = dep[:-1, :-1], dep[:-1, 1:], dep[1:, :-1], dep[1:, 1:]
    lim = EDGE * step * np.maximum(dtl, 0.5)
    tris = []
    for a, b, c, da, db, dc in ((tl, bl, tr, dtl, dbl, dtr), (tr, bl, br, dtr, dbl, dbr)):
        good = (a >= 0) & (b >= 0) & (c >= 0)
        good &= (np.abs(da - db) < lim) & (np.abs(da - dc) < lim) & (np.abs(db - dc) < lim)
        tris.append(np.stack([a[good], b[good], c[good]], 1))
    return np.concatenate(tris).astype(np.int32)


def read_ply(path):
    """Read the binary PLY written by kinect3d.py -> (xyz float32 Nx3, rgb uint8 Nx3, tris int32 Mx3)."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: not a PLY file")
            header += line
        text = header.decode()
        if "binary_little_endian" not in text:
            raise ValueError(f"{path}: only binary little-endian PLY from kinect3d.py is supported")
        nv = int(text.split("element vertex ")[1].split()[0])
        nf = int(text.split("element face ")[1].split()[0]) if "element face" in text else 0
        v = np.frombuffer(f.read(nv * 15), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                            ("r", "u1"), ("g", "u1"), ("b", "u1")])
        f_ = np.frombuffer(f.read(nf * 13), [("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    rgb = np.stack([v["r"], v["g"], v["b"]], 1)
    tris = np.stack([f_["a"], f_["b"], f_["c"]], 1) if nf else np.empty((0, 3), np.int32)
    return xyz, rgb, tris


def make_mesh(name, xyz, rgb, tris):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", xyz.ravel())
    if len(tris):
        me.loops.add(tris.size)
        me.polygons.add(len(tris))
        me.loops.foreach_set("vertex_index", tris.ravel())
        me.polygons.foreach_set("loop_start", np.arange(0, tris.size, 3, dtype=np.int32))
    me.update()
    col = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
    rgba = np.ones((len(rgb), 4), np.float32)
    rgba[:, :3] = rgb / 255.0
    col.data.foreach_set("color_srgb", rgba.ravel())
    me.color_attributes.active_color = col
    return me


def make_material():
    mat = bpy.data.materials.get("Kinect colour") or bpy.data.materials.new("Kinect colour")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emit = nt.nodes.new("ShaderNodeEmission")  # the colours are already lit by the real room
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    attr.location, emit.location, out.location = (-400, 0), (-150, 0), (100, 0)
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_points_nodes(mat, step):
    """Geometry Nodes: vertices -> visible dots whose radius grows with distance like the real pixel spacing."""
    ng = bpy.data.node_groups.new("Kinect points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    pos, sep = n.new("GeometryNodeInputPosition"), n.new("ShaderNodeSeparateXYZ")
    size = n.new("ShaderNodeValue")
    size.label = size.name = "Point size"
    size.outputs[0].default_value = POINT_SIZE * step / FX / 2  # radius = half the pixel gap at that depth
    mul = n.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    m2p, setmat = n.new("GeometryNodeMeshToPoints"), n.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    for node, x, y in ((gin, -700, 0), (pos, -700, -200), (sep, -500, -200), (size, -500, -380),
                       (mul, -300, -250), (m2p, -100, 0), (setmat, 100, 0), (gout, 300, 0)):
        node.location = (x, y)
    ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(pos.outputs[0], sep.inputs[0])
    ng.links.new(sep.outputs["Y"], mul.inputs[0])  # Y = distance from the Kinect
    ng.links.new(size.outputs[0], mul.inputs[1])
    ng.links.new(mul.outputs[0], m2p.inputs["Radius"])
    ng.links.new(m2p.outputs[0], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs[0], gout.inputs[0])
    return ng


def load(folder, save=None, mesh=MESH):
    folder = os.path.abspath(folder)
    files = sorted(glob.glob(os.path.join(folder, "frame_*.ply")))
    if not files:
        raise SystemExit(f"No frame_*.ply files in {folder}")
    meta = {}
    if os.path.exists(os.path.join(folder, "meta.json")):
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
    fps = meta.get("fps_actual")
    if not fps and len(files) > 1:  # older recordings: estimate from file times
        t = sorted(os.path.getmtime(p) for p in files)
        fps = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    fps = max(1, round(fps or 30))

    name = os.path.basename(folder)
    scene = bpy.context.scene
    for obj in [o for o in scene.objects if o.name in ("Cube", "Light", "Camera")]:
        bpy.data.objects.remove(obj)  # clear the default scene
    coll = bpy.data.collections.new(name)
    scene.collection.children.link(coll)

    mat = make_material()
    nodes = None
    first_xyz = None
    step = meta.get("step")
    print(f"Loading {len(files)} frames from {folder} at {fps} fps" + (" as surfaces" if mesh else "") + " ...")
    for i, path in enumerate(files, 1):
        xyz, rgb, tris = read_ply(path)
        if not step and len(xyz):
            step = detect_step(xyz)
        if mesh and not len(tris) and len(xyz):
            tris = grid_triangles(xyz, step)
        if nodes is None and not len(tris):
            nodes = make_points_nodes(mat, step or 1)
        if first_xyz is None:
            first_xyz = xyz
        me = make_mesh(f"{name}_{i:05d}", xyz, rgb, tris)
        obj = bpy.data.objects.new(me.name, me)
        coll.objects.link(obj)
        if len(tris):
            me.materials.append(mat)
        else:
            obj.modifiers.new("Points", "NODES").node_group = nodes
        # visible only on frame i
        for f, hidden in ((i - 1, True), (i, False), (i + 1, True)):
            obj.hide_viewport = obj.hide_render = hidden
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    scene.frame_start, scene.frame_end, scene.frame_current = 1, len(files), 1
    scene.render.fps, scene.render.fps_base = fps, 1
    scene.render.resolution_x, scene.render.resolution_y = W, H

    # Camera where the Kinect was, looking along +Y, same field of view
    cam_data = bpy.data.cameras.new("Kinect")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = 2 * math.atan(W / 2 / FX)
    cam_data.clip_start = 0.05
    cam = bpy.data.objects.new("Kinect", cam_data)
    cam.rotation_euler = (math.radians(90), 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.02, 0.02, 0.02)
    if scene.world.node_tree:
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)

    # Viewport: show colours and orbit around the middle of the scene
    centre = first_xyz.mean(0) if len(first_xyz) else (0, 2, 0)
    wm = bpy.context.window_manager
    for window in (wm.windows if wm else []):
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.shading.type = "MATERIAL"
                space.clip_start = 0.01
                r3d = space.region_3d
                r3d.view_location = centre
                r3d.view_distance = 3.0
                r3d.view_rotation = (cam.matrix_world.to_quaternion())
                r3d.view_perspective = "PERSP"

    if save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(save))
        print(f"Saved {os.path.abspath(save)}")
    print(f"Done: {len(files)} frames, {len(files) / fps:.1f}s at {fps} fps. Press Space to play.")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    folder, save = (argv[0] if argv else FOLDER), None
    if "--save" in argv:
        save = argv[argv.index("--save") + 1]
    if not folder:
        raise SystemExit("Give a recording folder: blender --python kinect_blender.py -- rec_XXXX  (or set FOLDER)")
    load(folder, save, mesh=MESH or "--mesh" in argv)


main()
