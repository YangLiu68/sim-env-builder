"""One command for all smoke tests of a MuJoCo environment.

    python -m simkit.smoke scene.xml --out smoke [--controller controller.py] [--task-checks task_checks.py]
                                                 [--seconds 4] [--size 960x540] [--expected-sizes sizes.json]

Writes into ``--out``:
  renders/*.png            every named camera + auto views (overview, top, front, side, robot closeups)
  contact_sheet.png        all views on one labelled sheet          <- LOOK AT THIS
  collision_sheet.png      same overview with collision geoms shown (are colliders where meshes are?)
  filmstrip.png            8 frames of a rollout with the controller   <- LOOK AT THIS
  rollout.mp4              the rollout as video (if ffmpeg is available)
  showcase.mp4             fly-around + demo controller + every named camera (skip with --no-showcase)
  render_report.json       per-image stats (blank/dark/washed out) + object visibility
  physics_report.json      physics examiner + layout checks
  smoke_report.json        everything above, summarised
  SMOKE.md                 human-readable summary + the visual checklist to fill in

Exit code 0 when nothing failed (warnings allowed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from . import physics_exam as PE
from .layout import check_layout
from .render import Views, actuated_roots, body_label, image_stats

VISUAL_CHECKLIST = """\
## Visual check (fill in after LOOKING at contact_sheet.png, filmstrip.png and closeups)

| # | Question | Answer (yes/no + note) |
|---|----------|------------------------|
| 1 | Is every object the user asked for visible and recognisable? | |
| 2 | Are real-world scales right (door ~2 m, person ~1.7 m, car ~4.5 m, table ~0.75 m)? | |
| 3 | Is everything upright, facing the right way, and resting on its support (no floating/sinking)? | |
| 4 | Do Poly Pizza models look right (correct model, colours, no missing/black parts)? | |
| 5 | Do procedural fallbacks read as the intended object? | |
| 6 | Lighting/sky/ground: well lit, no pitch-black or washed-out views? | |
| 7 | Filmstrip: does the robot/vehicle move the way the controller intends? | |
| 8 | Collision sheet: do colliders roughly match the visible shapes? | |
| 9 | Would the user say this matches their request? What is missing? | |
"""


def parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def run(scene, out, controller=None, task_checks=None, seconds=4.0, size=(960, 540), expected_sizes=None, allow_floating=(),
        showcase=True):
    out = Path(out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = {"scene": str(scene), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    # a controller may declare how long its scripted demo takes: DEMO_SECONDS = 12.0
    if controller:
        try:
            demo_s = getattr(PE.load_module(controller), "DEMO_SECONDS", None)
            if demo_s:
                seconds = float(min(max(float(demo_s), 1.0), 60.0))
                report["demo_seconds"] = seconds
        except Exception:  # noqa: BLE001 - a broken controller is reported by the examiner below
            pass

    # ---------------- physics
    phys = PE.examine(scene, controller, task_checks, seconds=min(seconds, 30.0))
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as e:  # noqa: BLE001
        report.update(physics=phys, render={"ok": False, "error": str(e)})
        (out / "smoke_report.json").write_text(json.dumps(report, indent=1, default=str))
        print("compile failed:", e)
        return report
    data = mujoco.MjData(model)
    PE.reset(model, data)
    try:
        es = json.loads(Path(expected_sizes).read_text()) if expected_sizes else None
        phys["checks"].extend(check_layout(model, data, es, allow_floating))
    except Exception as e:  # noqa: BLE001
        phys["checks"].append(PE.chk("layout_crashed", "layout", "fail", details=repr(e)))
    phys["summary"] = PE.summarize(phys["checks"])
    (out / "physics_report.json").write_text(json.dumps(phys, indent=1, default=str))

    # ---------------- rendering
    rend = {"images": {}, "problems": []}
    try:
        v = Views(model, *size)
        imgs = v.render_all(data)
        from PIL import Image

        for name, im in imgs.items():
            Image.fromarray(im).save(out / "renders" / f"{name}.png")
            st = image_stats(im)
            rend["images"][name] = st
            if not st["ok"]:
                rend["problems"].append(f"{name}: {', '.join(st['problems'])}")
        title = f"{Path(scene).parent.name}/{Path(scene).name}  |  {model.nbody - 1} bodies, {model.ngeom} geoms, {model.nu} actuators"
        v.contact_sheet(imgs, out / "contact_sheet.png", cols=3, title=title)
        # collision view of the overview
        auto = v.auto_views(data)
        col = {"visual_overview": imgs.get("auto_overview"), "collision_overview": v.render(data, auto["auto_overview"], show_collision=True)}
        roots = actuated_roots(model)
        if roots:
            k = f"auto_robot_{body_label(model, roots[0])}"
            if k in auto:
                col["visual_robot"] = imgs.get(k)
                col["collision_robot"] = v.render(data, auto[k], show_collision=True)
        v.contact_sheet({k: x for k, x in col.items() if x is not None}, out / "collision_sheet.png", cols=2)
        # visibility of objects (named cams + auto views)
        cams = dict(auto)
        cams.update(v.named_cameras())
        vis = v.visibility(data, cams)
        bodies = sorted({body_label(model, b) for b in range(1, model.nbody) if model.body_parentid[b] == 0})
        unseen = [b for b in bodies if vis.get(b, 0) < 30]
        rend["visibility_pixels"] = dict(sorted(vis.items(), key=lambda kv: -kv[1])[:60])
        rend["unseen_objects"] = unseen
        import re as _re0

        def _stem(n):
            return _re0.sub(r"[_-]?\d+$", "", n) or n
        seen_stems = {_stem(b) for b in bodies if b not in unseen}
        # a few hidden instances of a scattered family (rock03 behind a hill) are a hint; a requested
        # object nobody can see is a problem
        unseen_scatter = [b for b in unseen if _stem(b) in seen_stems]
        unseen_whole = [b for b in unseen if _stem(b) not in seen_stems]
        rend.setdefault("hints", [])
        if unseen_scatter:
            rend["hints"].append(f"{len(unseen_scatter)} instances of otherwise-visible objects are hidden in every view: {unseen_scatter[:12]}")
        if unseen_whole:
            rend["problems"].append(f"{len(unseen_whole)} top-level bodies never visible in any view: {unseen_whole[:12]}")
        # salience: objects that are visible but never bigger than a speck in ANY view
        import re as _re

        vmax = v.visibility(data, cams, reduce="max")
        tiny_px = max(150, int(0.0004 * size[0] * size[1]))
        tiny = [b for b in bodies if b not in unseen and vmax.get(b, 0) < tiny_px]
        groups: dict[str, list] = {}
        for b in tiny:
            groups.setdefault(_re.sub(r"[_-]?\d+$", "", b) or b, []).append(b)
        rend["barely_visible"] = {k: {"count": len(g), "max_px": max(vmax.get(x, 0) for x in g)} for k, g in groups.items()}
        rend.setdefault("hints", [])
        if tiny:  # a hint for the agent, not a defect: big scenes legitimately have small props
            rend["hints"].append("barely visible in every view (specks): " + ", ".join(
                f"{k} x{len(g)}" for k, g in list(groups.items())[:10]) + " -> bigger, closer camera, or intended?")
        # fixed (world) cameras: the robot at its home pose must not hide the TASK OBJECTS (movable,
        # non-robot bodies) - render each camera with and without the robot's geoms and compare
        robot_roots = set(actuated_roots(model))
        blocked, coverage = {}, {}
        if robot_roots:
            from .render import top_level_body

            geom_root = np.array([top_level_body(model, model.geom_bodyid[g]) for g in range(model.ngeom)])
            robot_geoms = np.isin(geom_root, list(robot_roots))
            movable = {b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and model.body_dofnum[b] > 0
                       and b not in robot_roots}
            task_geoms = np.isin(geom_root, list(movable)) if movable else np.zeros(model.ngeom, bool)

            def task_pixels(cid):
                seg = v.render(data, cid, segmentation=True)
                ids, types = seg[..., 0], seg[..., 1]
                ok = (types == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0) & (ids < model.ngeom)
                g = ids[ok]
                return int(task_geoms[g].sum()), float(robot_geoms[g].sum()) / ids.size
            for cid in range(model.ncam):
                if top_level_body(model, model.cam_bodyid[cid]) in robot_roots:
                    continue  # robot-mounted cameras are expected to see the robot
                cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) or str(cid)
                with_robot, frac = task_pixels(cid)
                coverage[cname] = round(frac, 3)
                if not task_geoms.any():
                    if frac > 0.25:
                        blocked[cname] = f"robot fills {int(100 * frac)}% of the frame"
                    continue
                saved = model.geom_group[robot_geoms].copy()
                model.geom_group[robot_geoms] = 5  # hidden group
                try:
                    without_robot, _ = task_pixels(cid)
                finally:
                    model.geom_group[robot_geoms] = saved
                if without_robot > 200:
                    hidden = 1 - with_robot / without_robot
                    if hidden > 0.2:
                        blocked[cname] = f"robot hides {int(100 * hidden)}% of the task objects"
        rend["robot_camera_coverage"] = coverage
        # robot-mounted navigation cameras (FPV, head, front) should not see mostly their own robot;
        # wrist/gripper cameras are meant to see the fingers
        if robot_roots:
            for cid in range(model.ncam):
                root = top_level_body(model, model.cam_bodyid[cid])
                cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) or str(cid)
                if root not in robot_roots or any(w in cname.lower() for w in ("wrist", "gripper", "hand", "palm", "ee")):
                    continue
                seg = v.render(data, cid, segmentation=True)
                ids, types = seg[..., 0], seg[..., 1]
                ok = (types == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0) & (ids < model.ngeom)
                own = float((geom_root[ids[ok]] == root).sum()) / ids.size
                if own > 0.15:
                    rend.setdefault("hints", []).append(f"onboard camera {cname} sees its own robot in {int(100 * own)}% of the image -> move it forward/up or hide those parts")
        busy = {k: f for k, f in coverage.items() if f > 0.08 and any(w in k.lower() for w in ("overhead", "top", "bird", "ceiling"))}
        if busy:
            rend.setdefault("hints", []).append("robot fills a large part of overhead camera(s) at home: " + ", ".join(
                f"{k} {int(100 * f)}%" for k, f in busy.items()) + " -> consider a home pose tucked out of view")
        rend["robot_blocks_camera"] = blocked
        if blocked:
            rend["problems"].append("robot at its home pose blocks fixed cameras: " + ", ".join(
                f"{k} ({v_})" for k, v_ in blocked.items()) + " -> move the camera or choose a home pose out of view")
        # rollout filmstrip
        ctrl_mod = PE.load_module(controller) if controller else None
        from .world import load_world, with_world

        ctrl = with_world(load_world(Path(scene).resolve().parent, model), PE.get_controller(model, ctrl_mod))
        d2 = mujoco.MjData(model)
        PE.reset(model, d2)
        def ctrl_factory():
            return with_world(load_world(Path(scene).resolve().parent, model), PE.get_controller(model, PE.load_module(controller) if controller else None))
        v.filmstrip(model, d2, ctrl, seconds=seconds, frames=8, path=out / "filmstrip.png", video_path=str(out / "rollout.mp4"),
                    controller_factory=ctrl_factory)
        rend["filmstrip"] = str(out / "filmstrip.png")
        rend["ok"] = not any("blank" in p for p in rend["problems"])
        v.close()
    except Exception as e:  # noqa: BLE001
        import traceback

        rend["ok"] = False
        rend["error"] = f"{e!r}\n{traceback.format_exc()[-1500:]}"
    # ---------------- showcase video (orbit + demo controller + every camera) for people to watch
    if showcase:
        try:
            from .showcase import render as render_showcase

            vid, dur = render_showcase(Path(scene).parent, out / "showcase.mp4", seconds=(seconds if "demo_seconds" in report else min(max(seconds * 2, 6.0), 14.0)),
                                       controller=controller, scene=Path(scene).name, name=Path(scene).resolve().parent.name)
            rend["showcase"] = {"path": str(vid), "seconds": round(dur, 1)}
        except Exception as e:  # noqa: BLE001 - a video is a nice-to-have, never a test failure
            rend["showcase"] = {"error": repr(e)}
    (out / "render_report.json").write_text(json.dumps(rend, indent=1, default=str))

    # ---------------- manifest for dashboards / RL users
    manifest = {
        "scene": Path(scene).name,
        "bodies": model.nbody - 1, "geoms": model.ngeom, "joints": model.njnt, "actuators": model.nu,
        "timestep": model.opt.timestep, "gravity": model.opt.gravity.tolist(),
        "cameras": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)],
        "actuator_list": [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}",
                           "ctrlrange": model.actuator_ctrlrange[a].tolist() if model.actuator_ctrllimited[a] else None}
                          for a in range(model.nu)],
        "keyframes": model.nkey,
    }
    (out / "env_manifest.json").write_text(json.dumps(manifest, indent=1))

    report.update(physics_summary=phys["summary"], render_ok=rend.get("ok"), render_problems=rend.get("problems"),
                  hints=rend.get("hints", []),
                  seconds=round(time.time() - t0, 1),
                  failed_checks=[f"{c['group']}/{c['name']}: {c['details'][:160]}" for c in phys["checks"] if c["status"] == "fail"],
                  warned_checks=[f"{c['group']}/{c['name']}: {json.dumps(c['value'], default=str)[:160]}" for c in phys["checks"] if c["status"] == "warn"])
    verdict = "FAIL" if (phys["summary"]["fail"] or not rend.get("ok")) else ("PASS_WITH_WARNINGS" if phys["summary"]["warn"] or rend.get("problems") else "PASS")
    report["verdict"] = verdict
    task = [c for c in phys["checks"] if c["group"] == "task"]
    exam = [c for c in phys["checks"] if c["group"] != "task"]
    cnt = lambda cs, st: sum(c["status"] == st for c in cs)  # noqa: E731
    report["summary_line"] = (f"Smoke {verdict}: examiner {cnt(exam, 'pass')} pass / {cnt(exam, 'warn')} warn / {cnt(exam, 'fail')} fail; "
                              f"task checks {cnt(task, 'pass')}/{len(task)} pass; render {'ok' if rend.get('ok') else 'PROBLEMS'}"
                              f"{' (' + str(len(rend.get('problems', []))) + ' render problems)' if rend.get('problems') else ''}")
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1, default=str))
    write_md(out, report, phys, rend)
    return report


def write_md(out, report, phys, rend):
    lines = [f"# Smoke test: {report['scene']}", "", f"**Verdict: {report['verdict']}**  ({report['seconds']} s)", "",
             f"Paste this line into REPORT.md: `{report.get('summary_line', '')}`", "",
             "## Physics examiner", "", "| status | group | check | value |", "|---|---|---|---|"]
    for c in phys["checks"]:
        lines.append(f"| {c['status'].upper()} | {c['group']} | {c['name']} | {json.dumps(c['value'], default=str)[:90]} |")
    lines += ["", "## Rendering", ""]
    if rend.get("error"):
        lines.append(f"Render error: `{rend['error'][:400]}`")
    for n, st in rend.get("images", {}).items():
        lines.append(f"- `{n}`: mean {st['mean']}, std {st['std']}, edges {st['edge_density']} {'OK' if st['ok'] else 'PROBLEM: ' + ', '.join(st['problems'])}")
    if rend.get("unseen_objects"):
        lines.append(f"- never visible: {rend['unseen_objects'][:20]}")
    lines += ["", "Images to inspect: `contact_sheet.png`, `filmstrip.png`, `collision_sheet.png`, `renders/`.", "", VISUAL_CHECKLIST]
    (out / "SMOKE.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--out", default="smoke")
    ap.add_argument("--controller")
    ap.add_argument("--task-checks")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--expected-sizes")
    ap.add_argument("--allow-floating", default="", help="comma-separated name fragments allowed to float (lamps, signs)")
    ap.add_argument("--no-showcase", action="store_true", help="skip the showcase video")
    a = ap.parse_args(argv)
    rep = run(a.scene, a.out, a.controller, a.task_checks, a.seconds, parse_size(a.size), a.expected_sizes,
              tuple(x for x in a.allow_floating.split(",") if x), showcase=not a.no_showcase)
    print(json.dumps({k: rep.get(k) for k in ("verdict", "summary_line", "physics_summary", "render_ok", "render_problems", "failed_checks", "warned_checks", "hints")}, indent=1, default=str))
    print(f"\nNow LOOK at {Path(a.out) / 'contact_sheet.png'} and {Path(a.out) / 'filmstrip.png'} and fill the checklist in {Path(a.out) / 'SMOKE.md'}.")
    return 0 if rep.get("verdict") != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
