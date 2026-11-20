"""README media for the flood_intersection golden task: the agent's own build (scene + world.py +
controller.py, unmodified) filmed around its surprise eruption, high overview + chase camera, English labels.

    ~/.cache/sim-env-builder/venv/bin/python harness/render_detour_video.py \
        golden/runs/iter_08/flood_intersection/work/oak_flood [--out docs/media/flood_intersection.mp4] [--gif-only]

Needs an environment whose scenario has a `Geyser("mh_surprise", start=...)` and whose controller logs
`{"event": "fountain", ...}` decisions (the iter_08 build does). Writes the .mp4 and a README .gif next to it.
"""
import subprocess
import sys
from pathlib import Path

ENV = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ENV))
import imageio.v2 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import controller as CT  # noqa: E402  (the agent's)
from simkit.render import apply_model_flags  # noqa: E402
from world import make_world  # noqa: E402  (the agent's)

W, H = 1280, 720
FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
BIG = ImageFont.truetype(FONT_PATH, 34)
MID = ImageFont.truetype(FONT_PATH, 26)
SMALL = ImageFont.truetype(FONT_PATH, 21)


class Run:
    def __init__(self):
        self.m = m = mujoco.MjModel.from_xml_path(str(ENV / "scene.xml"))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
        self.d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        mujoco.mj_forward(m, self.d)
        self.world = make_world(m)
        self.ctrl = CT.make_controller(m)
        self.spec, self.sc = self.world.spec, self.world.scenario
        self.gz = [g for g in self.spec.geysers if g.name == "mh_surprise"][0]
        self.L, self.Wd = self.sc["ego"].get("length", 4.7), self.sc["ego"]["width"]
        self.b = m.body("ego").id
        self.r = apply_model_flags(mujoco.Renderer(m, H, W), m)
        self.opt = mujoco.MjvOption()
        for i in range(6):
            self.opt.geomgroup[i] = 1 if i <= 2 else 0
        gx, gy = self.gz.pos                      # same framing relative to the fountain as the earlier demo
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (gx - 4.0, gy - 1.1, 0.0)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 24.0, 95.0, -50.0
        self.chase = mujoco.MjvCamera()
        self.chase.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.chase.trackbodyid = self.b
        self.chase.distance, self.chase.azimuth, self.chase.elevation = 12.0, 0.0, -24.0
        self.wet_stop = 0.0

    def advance(self, dt):
        for _ in range(int(round(dt / self.m.opt.timestep))):
            self.world(self.m, self.d)
            self.ctrl(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            if self.depth_under() > 0.02 and self.speed < 0.3:
                self.wet_stop += self.m.opt.timestep

    @property
    def speed(self):
        return float(np.linalg.norm(self.d.cvel[self.b][3:5]))

    def depth_under(self):
        p, R = self.d.xpos[self.b], self.d.xmat[self.b].reshape(3, 3)
        pts = [p[:2] + R[:2, :2] @ np.array([dx, dy]) for dx in (-self.L / 2, 0.0, self.L / 2) for dy in (-self.Wd / 2, 0.0, self.Wd / 2)]
        return max(float(self.spec.depth_at(q[0], q[1])) for q in pts)

    def views(self):
        self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        top = self.r.render().copy()
        self.r.update_scene(self.d, camera=self.chase, scene_option=self.opt)
        return top, self.r.render().copy()

    def decision(self):
        ev = [e for e in self.ctrl.log if e.get("event") == "fountain" and e.get("fountain") == "mh_surprise"]
        if not ev:
            return None, None
        e = ev[0]
        if e.get("offset") is None:
            return "Decision: no passable side, stop before the water", e
        other = [c for c in e["candidates"] if c.get("side") != e["side"]]
        why = ""
        if other and other[0].get("blocked"):
            bl, o = str(other[0]["blocked"]), other[0]["side"]
            why = (f"{o} side would hit the curb" if ("edge" in bl or "kerb" in bl) else
                   f"{o} side blocked by a car" if ("obstacle" in bl or "mover" in bl) else
                   f"{o} side too deep" if "deep" in bl else f"{o} side not passable")
        return f"Decision: pass on the {e['side']} by {abs(e['offset']):.1f} m" + (f" ({why})" if why else ""), e


def tag(dr, x, y, text, font, fill=(0, 0, 0)):
    w = dr.textlength(text, font=font)
    h = font.size + 12
    dr.rectangle((x, y, x + w + 20, y + h), fill=fill)
    dr.text((x + 10, y + 4), text, fill=(255, 255, 255), font=font)
    return y + h + 6


def frame(run):
    top, chase = run.views()
    im = Image.fromarray(np.concatenate([top, chase], 0))
    im.paste((255, 255, 255), (0, H - 1, W, H + 1))
    dr = ImageDraw.Draw(im)
    t = run.d.time
    y = tag(dr, 12, 12, "Live detour", BIG, fill=(30, 110, 60))
    dec, e = run.decision()
    if e is not None and t < e["t"] + 2.5:
        y = tag(dr, 12, y, f"Manhole erupts in our lane, {e['distance']:.0f} m ahead", MID, fill=(170, 40, 30))
    if dec:
        tag(dr, 12, y, dec, MID, fill=(30, 110, 60))
    yb = H + 12
    for line in (f"Speed {run.speed * 3.6:4.1f} km/h",
                 f"Max water under car {run.depth_under():.2f} m (safe ≤ {run.sc['safe_wading']:.2f} m)",
                 f"Time stopped in water {run.wet_stop:.1f} s"):
        yb = tag(dr, 12, yb, line, MID)
    tag(dr, 12, 2 * H - SMALL.size - 24, "Scene, physics and driving logic generated by an AI agent (unmodified)", SMALL)
    tag(dr, W - 150, 2 * H - 52, f"t = {t:4.1f} s", MID)
    return np.asarray(im)


def gif(video: Path, out: Path, t0: float = 2.4, t1: float = 10.4, width: int = 420, fps: int = 8):
    """The eruption and the pass, for the README (portrait, like the video)."""
    import imageio_ffmpeg

    flt = (f"[0:v]trim={t0}:{t1},setpts=PTS-STARTPTS,fps={fps},scale={width}:-1:flags=lanczos,split[x][y];"
           f"[x]palettegen=max_colors=160:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(video), "-filter_complex", flt, str(out)],
                   check=True)


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else str(Path(__file__).resolve().parents[1] / "docs/media/flood_intersection.mp4")
    if "--gif-only" in sys.argv:
        gif(Path(out), Path(out).with_suffix(".gif"))
        print("wrote", Path(out).with_suffix(".gif"))
        return
    run = Run()
    t_erupt = run.gz.start
    if "--still" in sys.argv:
        for k, ts in enumerate(float(v) for v in sys.argv[sys.argv.index("--still") + 1].split(",")):
            run.advance(ts - run.d.time)
            Image.fromarray(frame(run)).save(out.replace(".png", f"_{k}.png"))
        return
    fps = 30
    t0, t_end = t_erupt - 3.0, t_erupt + 13.0
    run.advance(t0)
    w = iio.get_writer(out, fps=fps, codec="libx264", quality=None, ffmpeg_params=["-crf", "23", "-preset", "medium"],
                       pixelformat="yuv420p", macro_block_size=8)
    for _ in range(int((t_end - t0) * fps)):
        run.advance(1 / fps)
        w.append_data(frame(run))
    w.close()
    gif(Path(out), Path(out).with_suffix(".gif"))
    print("wrote", out, "wet_stop", round(run.wet_stop, 2), "decision", run.decision()[0])


if __name__ == "__main__":
    main()
