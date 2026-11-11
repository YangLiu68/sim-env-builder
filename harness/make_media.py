"""Build README media from golden runs: hero GIFs, compressed MP4s and a gallery of every task.

    .venv/bin/python harness/make_media.py            # writes docs/media/

GIF = 3 s of the showcase fly-around + 3 s of the demo controller in action (480 px, 12 fps).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as iio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "golden/runs"
OUT = ROOT / "docs/media"
FF = imageio_ffmpeg.get_ffmpeg_exe()

HERO = [  # (iteration, task, caption)
    ("iter_05", "isaac_synthetic", "Retail aisle for synthetic data (Isaac Sim export + MuJoCo)"),
    ("iter_05", "sd_intersection", "Self-driving: city intersection with lights and pedestrians"),
    ("iter_05", "mars_rover", "Mars rover on cratered terrain, 3.71 m/s² gravity"),
    ("iter_03", "factory_sorting", "Factory cell: conveyor + arm sorting by colour"),
    ("iter_02", "warehouse_amr", "Warehouse AMR driving an aisle to its goal"),
    ("iter_04", "home_fetch", "Home robot fetching a cup in a living room"),
    ("iter_04", "orchard_drone", "Orchard inspection rover between apple-tree rows"),
    ("iter_03", "robot_soccer", "Multi-agent robot soccer, 2 vs 2"),
]


def score_of(it: str, task: str):
    r = json.loads((RUNS / it / task / "result.json").read_text())
    return r.get("total_score")


def video_of(it: str, task: str) -> Path:
    return RUNS / it / task / "regrade/showcase.mp4"


def demo_seconds(video: Path) -> float:
    """Showcase = 4 s orbit, then the demo, then 2.5 s per named camera."""
    meta = iio.get_reader(str(video)).get_meta_data()
    return float(meta.get("duration", 12.0))


def gif(video: Path, out: Path, width=480, fps=12):
    dur = demo_seconds(video)
    a0, a1 = 0.6, 3.6                                   # fly-around
    b0 = min(4.0 + max(dur - 4.0, 4.0) * 0.35, dur - 3.2)  # action, a third into the demo
    b1 = b0 + 3.0
    flt = (f"[0:v]trim={a0}:{a1},setpts=PTS-STARTPTS[a];[0:v]trim={b0}:{b1},setpts=PTS-STARTPTS[b];"
           f"[a][b]concat=n=2:v=1:a=0,fps={fps},scale={width}:-1:flags=lanczos,split[x][y];"
           f"[x]palettegen=max_colors=160:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([FF, "-loglevel", "error", "-y", "-i", str(video), "-filter_complex", flt, str(out)], check=True)


def mp4(video: Path, out: Path):
    subprocess.run([FF, "-loglevel", "error", "-y", "-i", str(video), "-vf", "scale=960:-2", "-c:v", "libx264", "-crf", "30",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)


def gallery(out: Path, tile=(400, 225), cols=4):
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    board = json.loads((ROOT / "golden/scoreboard.json").read_text())
    best = {}
    for it in board["iterations"]:
        if it.get("void"):
            continue
        for r in it["rows"]:
            if r["total"] is not None and (r["task"] not in best or r["total"] >= best[r["task"]][0]):
                best[r["task"]] = (r["total"], it["iter"], r["category"])
    items = sorted(best.items(), key=lambda kv: -kv[1][0])
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (tile[0] * cols, tile[1] * rows), (18, 20, 26))
    d = ImageDraw.Draw(sheet)
    for k, (task, (score, it, cat)) in enumerate(items):
        v = video_of(it, task)
        r = iio.get_reader(str(v))
        n = r.count_frames()
        frame = Image.fromarray(r.get_data(int(n * 0.4))).resize(tile)
        x, y = (k % cols) * tile[0], (k // cols) * tile[1]
        sheet.paste(frame, (x, y))
        d.rectangle([x, y + tile[1] - 40, x + tile[0], y + tile[1]], fill=(10, 12, 18))
        d.text((x + 8, y + tile[1] - 37), task, fill=(240, 244, 250), font=font)
        d.text((x + 8, y + tile[1] - 18), f"{cat}  ·  score {score}", fill=(160, 175, 200), font=small)
    sheet.save(out, quality=85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for it, task, _ in HERO:
        v = video_of(it, task)
        gif(v, OUT / f"{task}.gif")
        mp4(v, OUT / f"{task}.mp4")
        print(f"{task}: gif {(OUT / f'{task}.gif').stat().st_size / 1e6:.1f} MB, mp4 {(OUT / f'{task}.mp4').stat().st_size / 1e6:.1f} MB")
    gallery(OUT / "all_tasks.jpg")
    print("gallery", (OUT / "all_tasks.jpg").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    sys.exit(main())
