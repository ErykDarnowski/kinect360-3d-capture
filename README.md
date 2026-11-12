# Kinect 3D capture

Capture 3D scans and depth recordings with an Xbox 360 Kinect on Linux, then play them back in Blender.

| File | What it does |
|---|---|
| `kinect3d.py` | Captures depth + colour from the Kinect and saves it as 3D `.ply` files (single scans or sequences) |
| `kinect_blender.py` | Loads a recorded sequence into Blender as an animation you can play back and orbit around |

## Quick start

```bash
freenect-glview                                             # 1. live preview: check what's in view
python3 kinect3d.py record --fps 30 --step 2                # 2. record, Ctrl+C to stop -> rec_<date>_<time>/
blender --python kinect_blender.py -- rec_<date>_<time>     # 3. play it back in 3D (Space)
```

## Setup (Fedora)

**Hardware.** The Kinect needs its 12 V power adapter plugged into the wall as well as USB. Check that all three parts are detected:

```bash
lsusb | grep 045e    # should list Xbox NUI Motor, Audio and Camera
```

If only the **Motor** shows up, the camera isn't getting 12 V. Check the adapter's wall plug and barrel connector.

**Driver.** Install [libfreenect](https://github.com/OpenKinect/libfreenect) and stop the kernel's basic webcam driver from claiming the camera:

```bash
sudo dnf install libfreenect libfreenect-opencv
sudo modprobe -r gspca_kinect
echo "blacklist gspca_kinect" | sudo tee /etc/modprobe.d/blacklist-kinect.conf
```

**Permissions.** Let the logged-in user use the Kinect without root:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02b0|02ad|02ae", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/70-kinect-uaccess.rules
sudo udevadm control --reload && sudo udevadm trigger
```

Then unplug and replug the Kinect's USB.

**Python.** The scripts need only `numpy`. Blender comes with its own copy.

## Live viewers

These come with libfreenect. **Only one program can use the camera at a time**, so close them before recording.

| Command | Shows |
|---|---|
| `freenect-glview` | Depth + RGB side by side. `i` switches to the raw IR camera, `w`/`x` tilt the motor |
| `freenect-glpclview` | Live 3D point cloud, rotate with the mouse |
| `freenect-regview` | Depth aligned to RGB |
| `freenect-tiltdemo` | Accelerometer readings and tilt motor |

## Recording: `kinect3d.py`

```bash
python3 kinect3d.py snap                    # one scan -> scan_<time>.ply (median of 5 frames, less noise)
python3 kinect3d.py snap --mesh             # one scan as a solid surface
python3 kinect3d.py record --fps 30         # sequence -> rec_<time>/frame_00001.ply ... (Ctrl+C to stop)
```

| Option | Default | Meaning |
|---|---|---|
| `--mesh` | off | Join neighbouring points into triangles (files are about 3× larger) |
| `--step N` | 1 | Use every Nth pixel. `2` = 320×240, 4× smaller files |
| `--max-dist M` | 4.0 | Drop anything further than M metres. Use e.g. `1.2` to cut the background |
| `--fps N` | 5 | Target frame rate for `record` |
| `--avg N` | 5 | `snap`: number of frames to median-filter |
| `--tilt DEG` | — | Tilt the motor (-27…27°) before capturing |
| `-o PATH` | auto | Output file (`snap`) or folder (`record`) |

**Maximum frame rates** (measured; the sensor itself delivers 30 fps):

| Settings | fps | Size per frame |
|---|---|---|
| points, full resolution | ~17 | 3.3 MB |
| points, `--step 2` | 30 | 0.8 MB |
| mesh, full resolution | ~9 | 9.0 MB |
| mesh, `--step 2` | 30 | 2.2 MB |

Each recording folder also contains a `meta.json` with the actual fps, step and other settings. The Blender script reads it.

### Output format

Binary little-endian PLY: `x y z` (float, metres) + `red green blue` (uchar), plus triangle faces with `--mesh`.
Coordinates follow Blender's convention, with the Kinect at the origin:

- **X** right
- **Y** away from the Kinect (depth)
- **Z** up

Depth is captured in *registered* mode (mm, aligned to the RGB camera). It is converted to 3D with a pinhole model using typical Kinect 360 intrinsics (`fx 529.2, fy 525.6, cx 328.9, cy 267.5`), not a per-unit calibration.

## Playback in Blender: `kinect_blender.py`

```bash
blender --python kinect_blender.py -- rec_20261111_174146                         # open as points
blender --python kinect_blender.py -- rec_20261111_174146 --mesh                  # join points into surfaces
blender --python kinect_blender.py -- rec_20261111_174146 --save scene.blend      # also save a .blend
blender -b --python kinect_blender.py -- rec_20261111_174146 --save scene.blend   # build .blend without a window
```

Each frame becomes its own object, visible only on its own frame of the timeline.

- **Space**: play / pause
- **Middle-mouse drag**: orbit while it plays
- **Numpad 0**: view through the `Kinect` camera (placed where the real sensor was, same ~62° field of view)
- **Dot size**: Geometry Nodes tab → *Kinect points* group → *Point size*
- **Render a video**: Output Properties → FFmpeg Video, then Render → Render Animation

`--mesh` works on point-cloud recordings too. It projects each point back to its Kinect pixel and joins neighbouring pixels. Points are not joined across depth jumps larger than `EDGE` (top of the script, default 4%), so foreground objects aren't webbed to the background. Lower it if gaps between close objects fill in; raise it if surfaces look torn.

## Limitations

- **Minimum distance about 0.8 m.** Anything closer has no depth data.
- **Depth noise of a few mm**, growing with distance. Small details close together may merge.
- **Single viewpoint.** Objects hide what's behind them, and every edge has a *depth shadow*: a strip with no data, because the IR projector sits a few cm beside the camera.
- **Surfaces are one-sided.** Seen from the side or behind, a scan is a shell with holes.

## Tips

- Keep the subject about 0.8–1.5 m from the Kinect for the best detail.
- Use full resolution (no `--step`) when detail matters more than frame rate.
- Use `--max-dist` to drop the background and keep files small.
- Keep the Kinect still while recording. Each frame is in the sensor's own coordinates, so moving it shifts the whole scene.

## License

[GPL-3.0-or-later](LICENSE). You can use, change and share this code, but anything you distribute that includes it must also be open source under the GPL.
