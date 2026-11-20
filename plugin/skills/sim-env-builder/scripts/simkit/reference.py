"""Reference media: recreate a scene the user showed you in photos or videos.

    # 1. keyframes you can LOOK at (videos are sampled evenly; photos are copied)
    python -m simkit.reference extract reference/ --out reference/frames --per-video 8
        -> reference/frames/<name>_<k>.jpg, reference/frames/index.json, reference/sheet.jpg  <- LOOK AT THIS

    # 2. after building: the media's viewpoint vs your camera, side by side + 50 % overlay
    python -m simkit.reference compare scene.xml --camera cam_reference --frame reference/frames/oak_flood_1_02.jpg \
        --out smoke/reference_compare.png [--seconds 3]

    # 3. make the smoke test produce the comparison every run: reference/match.json
    [{"camera": "cam_reference", "frame": "reference/frames/oak_flood_1_02.jpg", "seconds": 0}]

The comparison renders the camera at the reference frame's aspect ratio (portrait phone video stays
portrait), with world.py dynamics and the demo controller run for ``seconds`` first, so moving
things are where the media shows them. Use the overlay to line up horizon, kerb lines and the big
objects; adjust the camera's pos/target/fovy (phone main camera: ~65-70 deg along the long side).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".gif"}


def _font(size):
    from .render import _font as f

    return f(size)


def _video_frames(path: Path, n: int):
    """[(t_seconds, HxWx3 uint8)] n frames evenly spread over the video."""
    import imageio.v2 as iio

    r = iio.get_reader(str(path))
    meta = r.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    dur = float(meta.get("duration") or 0.0)
    if not dur:
        try:
            dur = r.count_frames() / fps
        except Exception:  # noqa: BLE001
            dur = 10.0
    out = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        try:
            out.append((t, np.asarray(r.get_data(int(t * fps)))[..., :3]))
        except Exception:  # noqa: BLE001 - some containers do not seek; fall back to sequential reading
            break
    if len(out) < n:  # sequential fallback
        r.close()
        r = iio.get_reader(str(path))
        want = {int(dur * (k + 0.5) / n * fps) for k in range(n)}
        out = [(i / fps, np.asarray(f)[..., :3]) for i, f in enumerate(r) if i in want]
    r.close()
    return out


def extract(src, out="reference/frames", per_video: int = 8, max_side: int = 960, sheet: str | None = None) -> list[dict]:
    """Keyframes from every photo/video in ``src`` (file or folder) -> ``out``; writes index.json and a
    labelled sheet (default: <out>/../sheet.jpg). Returns [{"frame", "source", "t", "size"}]."""
    from PIL import Image

    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (src.iterdir() if src.is_dir() else [src])
                   if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT and out.resolve() not in p.resolve().parents)
    index = []
    for f in files:
        items = ([(None, np.asarray(Image.open(f).convert("RGB")))] if f.suffix.lower() in IMAGE_EXT
                 else _video_frames(f, per_video))
        for k, (t, img) in enumerate(items):
            im = Image.fromarray(img)
            s = max_side / max(im.size)
            if s < 1:
                im = im.resize((int(im.size[0] * s), int(im.size[1] * s)))
            name = f"{f.stem}_{k:02d}.jpg" if t is not None else f"{f.stem}.jpg"
            im.save(out / name, quality=90)
            index.append({"frame": str(out / name), "source": f.name, "t": None if t is None else round(t, 2),
                          "size": list(im.size)})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    if index:
        contact(index, Path(sheet) if sheet else out.parent / "sheet.jpg")
    return index


def contact(index: list[dict], path: Path, tile_h: int = 420, cols: int = 6):
    """One labelled sheet of all keyframes (name @ time)."""
    from PIL import Image, ImageDraw

    ims = [Image.open(e["frame"]).convert("RGB") for e in index]
    ims = [im.resize((max(1, int(im.size[0] * tile_h / im.size[1])), tile_h)) for im in ims]
    tile_w = max(im.size[0] for im in ims)
    cols = min(cols, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 24)), (20, 20, 24))
    dr = ImageDraw.Draw(sheet)
    for i, (im, e) in enumerate(zip(ims, index)):
        x, y = (i % cols) * tile_w, (i // cols) * (tile_h + 24)
        sheet.paste(im, (x + (tile_w - im.size[0]) // 2, y + 24))
        label = Path(e["frame"]).name + (f" @ {e['t']:.1f}s" if e.get("t") is not None else "")
        dr.text((x + 6, y + 4), label, fill=(235, 235, 235), font=_font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)
    return path


def render_camera(scene, camera: str, width: int, height: int, seconds: float = 0.0, controller=None):
    """Render ``camera`` of scene.xml at width x height after ``seconds`` of world dynamics + controller."""
    import mujoco

    from . import physics_exam as PE
    from .render import apply_model_flags
    from .world import load_world, with_world

    scene = Path(scene)
    m = mujoco.MjModel.from_xml_path(str(scene))
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    d = mujoco.MjData(m)
    PE.reset(m, d)
    if seconds > 0:
        ctrl_mod = PE.load_module(controller) if controller else None
        step = with_world(load_world(scene.resolve().parent, m), PE.get_controller(m, ctrl_mod))
        for _ in range(int(seconds / m.opt.timestep)):
            if step:
                step(m, d)
            mujoco.mj_step(m, d)
    else:
        w = load_world(scene.resolve().parent, m)  # place mocap props (foam, NPCs) where world.py puts them at t=0
        if w:
            w(m, d)
        mujoco.mj_forward(m, d)
    r = apply_model_flags(mujoco.Renderer(m, height, width), m)
    opt = mujoco.MjvOption()
    for i in range(6):
        opt.geomgroup[i] = 1 if i <= 2 else 0
    r.update_scene(d, camera=camera, scene_option=opt)
    img = r.render().copy()
    r.close()
    return img


def compare(scene, camera: str, frame, out, seconds: float = 0.0, controller=None, height: int = 720) -> Path:
    """[reference | render | 50 % overlay] at the reference's aspect ratio -> ``out``."""
    from PIL import Image, ImageDraw

    ref = Image.open(frame).convert("RGB")
    aspect = ref.size[0] / ref.size[1]
    h = int(height)
    w = int(round(h * aspect / 2) * 2)
    if w > 1600:  # wide panoramas
        w, h = 1600, int(1600 / aspect)
    ren = Image.fromarray(render_camera(scene, camera, w, h, seconds, controller))
    ref = ref.resize((w, h))
    blend = Image.blend(ref, ren, 0.5)
    sheet = Image.new("RGB", (3 * w, h + 30), (20, 20, 24))
    dr = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate([(ref, f"reference: {Path(frame).name}"), (ren, f"render: {camera}" + (f" @ {seconds:g}s" if seconds else "")),
                                     (blend, "overlay 50/50")]):
        sheet.paste(im, (i * w, 30))
        dr.text((i * w + 8, 7), label, fill=(235, 235, 235), font=_font(16))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def compare_from_match(env_dir, out, controller=None) -> Path | None:
    """Smoke hook: every entry of <env>/reference/match.json, stacked into one image."""
    from PIL import Image

    env_dir = Path(env_dir)
    mf = env_dir / "reference" / "match.json"
    if not mf.exists():
        return None
    entries = json.loads(mf.read_text())
    entries = entries if isinstance(entries, list) else [entries]
    rows = []
    for k, e in enumerate(entries):
        tmp = Path(out).with_name(f"_refcmp_{k}.png")
        compare(env_dir / e.get("scene", "scene.xml"), e["camera"], env_dir / e["frame"], tmp,
                seconds=float(e.get("seconds", 0)), controller=controller, height=int(e.get("height", 640)))
        rows.append(Image.open(tmp).convert("RGB"))
        tmp.unlink()
    W = max(r.size[0] for r in rows)
    sheet = Image.new("RGB", (W, sum(r.size[1] for r in rows)), (20, 20, 24))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.size[1]
    sheet.save(out)
    return Path(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("src")
    e.add_argument("--out", default="reference/frames")
    e.add_argument("--per-video", type=int, default=8)
    e.add_argument("--max-side", type=int, default=960)
    c = sub.add_parser("compare")
    c.add_argument("scene")
    c.add_argument("--camera", required=True)
    c.add_argument("--frame", required=True)
    c.add_argument("--out", default="smoke/reference_compare.png")
    c.add_argument("--seconds", type=float, default=0.0)
    c.add_argument("--controller")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        idx = extract(a.src, a.out, a.per_video, a.max_side)
        print(f"{len(idx)} frames -> {a.out}; sheet: {Path(a.out).parent / 'sheet.jpg'}  <- LOOK at it")
    else:
        p = compare(a.scene, a.camera, a.frame, a.out, a.seconds, a.controller)
        print(f"wrote {p}  <- LOOK: do horizon, kerbs and the big objects line up?")


if __name__ == "__main__":
    main()
