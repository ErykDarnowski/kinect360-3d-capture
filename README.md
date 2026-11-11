# README

1. Preview and control

```bash
freenect-glview
```

2. Record depth video

```bash
py kinect3d.py record --fps 30 --step 2
```

3. View depth video (in **Blender**)

```bash
blender --python kinect_blender.py -- <dir_name> --save <file_name>.blend
```
