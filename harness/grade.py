"""Automated grading of one golden run (0-40 points). The judge adds 0-60.

    python harness/grade.py golden/runs/iter_01/drone_gates

Independent of the agent's own claims: rebuilds the env from build_env.py in a clean copy,
re-runs the smoke suite there, runs env.py, inspects assets and the transcript.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SKILL, find_env_dir, read_json, write_json  # noqa: E402

DELIVERABLES = ["SPEC.md", "README.md", "REPORT.md", "build_env.py", "scene.xml", "env.py", "controller.py", "task_checks.py",
                "assets/ASSETS.md", "smoke/smoke_report.json", "smoke/visual_check.md", "smoke/contact_sheet.png", "smoke/filmstrip.png"]


def sim_python() -> str:
    return subprocess.run(["bash", str(SKILL / "scripts/setup_env.sh")], capture_output=True, text=True, timeout=900).stdout.strip().splitlines()[-1]


def sh(cmd, cwd, timeout):
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return -9, f"timeout after {timeout}s"


def transcript_stats(path: Path) -> dict:
    tools, images_read, pp_calls, errors = {}, [], 0, 0
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if c.get("type") == "tool_use":
                    name = c.get("name", "")
                    tools[name] = tools.get(name, 0) + 1
                    if name.startswith("mcp__polypizza"):
                        pp_calls += 1
                    if name == "Read" and str((c.get("input") or {}).get("file_path", "")).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        images_read.append(Path(c["input"]["file_path"]).name)
        if ev.get("type") == "user":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    errors += 1
    return {"tool_calls": tools, "images_read": images_read, "polypizza_calls": pp_calls, "tool_errors": errors,
            "total_tool_calls": sum(tools.values())}


def grade(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run_meta.json", {}) or {}
    work = run_dir / "work"
    env = Path(meta["env_dir"]) if meta.get("env_dir") else find_env_dir(work)
    g: dict = {"env_dir": str(env) if env else None}
    ts = transcript_stats(run_dir / "transcript.jsonl")
    g["transcript"] = ts
    if env is None:
        g.update(auto_score=0.0, fatal="no scene.xml produced")
        return g
    # only require deliverables the playbook version used for this run actually asked for
    snap = run_dir.parent / "plugin_snapshot/skills/sim-env-builder/SKILL.md"
    playbook = snap.read_text() if snap.exists() else (SKILL / "SKILL.md").read_text()
    wanted = [d for d in DELIVERABLES if Path(d).name in playbook or d in ("scene.xml", "smoke/smoke_report.json")]
    present = {d: (env / d).exists() for d in wanted}
    g["deliverables"] = present
    pts = {"deliverables": round(8 * sum(present.values()) / len(present), 2)}
    pts["confirmation"] = 4.0 if (meta.get("asked_confirmation") and not meta.get("built_before_confirm")) else 0.0

    py = sim_python()
    tmp = Path(tempfile.mkdtemp(prefix="regrade_"))
    copy = tmp / env.name
    shutil.copytree(env, copy, ignore=shutil.ignore_patterns("smoke", "__pycache__", "*.mp4"))
    old_scene = (copy / "scene.xml").read_text() if (copy / "scene.xml").exists() else ""
    rc, log = sh([py, "build_env.py"], copy, 900) if (copy / "build_env.py").exists() else (1, "no build_env.py")
    compiled = False
    if rc == 0:
        rc2, log2 = sh([py, "-c", "import mujoco,sys; mujoco.MjModel.from_xml_path('scene.xml'); print('ok')"], copy, 120)
        compiled = rc2 == 0
    g["rebuild"] = {"ok": rc == 0 and compiled, "identical_scene": (copy / "scene.xml").exists() and (copy / "scene.xml").read_text() == old_scene,
                    "log_tail": log[-1200:]}
    pts["rebuild"] = 4.0 if g["rebuild"]["ok"] else 0.0
    if not (copy / "scene.xml").exists():  # rebuild failed: still regrade the agent's scene
        shutil.copy(env / "scene.xml", copy / "scene.xml")
    args = [py, "-m", "simkit.smoke", "scene.xml", "--out", "smoke_regrade", "--size", "960x540"]
    if (copy / "controller.py").exists():
        args += ["--controller", "controller.py"]
    if (copy / "task_checks.py").exists():
        args += ["--task-checks", "task_checks.py"]
    rc, log = sh(args, copy, 1500) if (copy / "simkit").exists() else (1, "no vendored simkit")
    rep = read_json(copy / "smoke_regrade/smoke_report.json", {}) or {}
    phys = read_json(copy / "smoke_regrade/physics_report.json", {}) or {}
    verdict = rep.get("verdict", "ERROR")
    task_checks = [c for c in phys.get("checks", []) if c.get("group") == "task"]
    g["smoke_regrade"] = {"verdict": verdict, "failed": rep.get("failed_checks"), "warned": rep.get("warned_checks"),
                          "render_problems": rep.get("render_problems"), "task_checks": [(c["name"], c["status"]) for c in task_checks],
                          "log_tail": log[-1500:] if verdict == "ERROR" else ""}
    agent_rep = read_json(env / "smoke/smoke_report.json", {}) or {}
    g["agent_smoke_verdict"] = agent_rep.get("verdict")
    g["honest_smoke"] = agent_rep.get("verdict") == verdict or (agent_rep.get("verdict") == "FAIL")
    pts["smoke"] = {"PASS": 10.0, "PASS_WITH_WARNINGS": 7.0}.get(verdict, 0.0)
    if task_checks:
        frac = sum(c["status"] == "pass" for c in task_checks) / len(task_checks)
        pts["task_checks"] = round(4 * frac * min(1.0, len(task_checks) / 2), 2)
    else:
        pts["task_checks"] = 0.0
    # keep regrade images for the judge and the dashboard
    dst = run_dir / "regrade"
    if dst.exists():
        shutil.rmtree(dst)
    if (copy / "smoke_regrade").exists():
        shutil.copytree(copy / "smoke_regrade", dst, ignore=shutil.ignore_patterns("rollout.mp4"))
    # showcase video for people (dashboard): rendered with the live toolkit if the env's own copy has none
    if not (dst / "showcase.mp4").exists():
        args = [py, str(ROOT / "harness/showcase.py"), str(copy), "--out", str(dst / "showcase.mp4"), "--seconds", "12"]
        if (copy / "controller.py").exists():
            args += ["--controller", str(copy / "controller.py")]
        dst.mkdir(parents=True, exist_ok=True)
        rc, log = sh(args, copy, 900)
        g["showcase"] = {"ok": rc == 0 and (dst / "showcase.mp4").exists(), "log_tail": log[-400:] if rc else ""}
    # latest examiner cross-check (catches problems an older vendored simkit missed)
    rc, log = sh([py, "-m", "simkit.physics_exam", "scene.xml", "--seconds", "2", "--out", "latest_exam.json"],
                 copy, 600) if True else (1, "")
    cross = read_json(copy / "latest_exam.json", {}) or {}
    g["latest_examiner"] = cross.get("summary")
    # env.py
    rc, log = sh([py, "env.py"], copy, 600) if (copy / "env.py").exists() else (1, "no env.py")
    g["env_py"] = {"ok": rc == 0, "log_tail": log[-800:]}
    pts["env_py"] = 4.0 if rc == 0 else 0.0
    # visual check evidence
    vc = (env / "smoke/visual_check.md").read_text() if (env / "smoke/visual_check.md").exists() else ""
    rows = len(re.findall(r"^\|\s*\d+\s*\|", vc, re.M))
    looked = any(n.startswith(("contact_sheet", "filmstrip")) for n in ts.get("images_read", []))
    g["visual_check"] = {"rows": rows, "looked_at_sheets": looked, "images_read": ts.get("images_read", [])[:40]}
    pts["visual_check"] = 3.0 if (rows >= 6 and looked) else (1.5 if (rows or looked) else 0.0)
    # assets
    pp_dir = env / "assets/polypizza"
    glbs = list(pp_dir.glob("*.glb")) if pp_dir.exists() else []
    manifest = read_json(pp_dir / "polypizza_assets.json", {}) or {}
    g["assets"] = {"polypizza_models": len(glbs), "manifest_entries": len(manifest), "credits": (pp_dir / "CREDITS.md").exists(),
                   "assets_md": (env / "assets/ASSETS.md").exists(), "polypizza_calls": ts.get("polypizza_calls", 0),
                   "procedural_meshes": len(list((env / "assets/procedural").glob("*.obj"))) if (env / "assets/procedural").exists() else 0}
    a = g["assets"]
    amd = (env / "assets/ASSETS.md").read_text(errors="replace").lower() if a["assets_md"] else ""
    no_models_needed = a["polypizza_models"] == 0 and any(w in amd for w in ("procedural", "primitive", "no poly pizza", "not needed", "built in code"))
    searched_ok = a["polypizza_calls"] > 0 and (a["polypizza_models"] == 0 or a["credits"])
    pts["assets"] = 3.0 if (a["assets_md"] and (searched_ok or no_models_needed)) else (1.0 if a["assets_md"] else 0.0)
    g["points"] = pts
    g["auto_score"] = round(sum(pts.values()), 2)
    shutil.rmtree(tmp, ignore_errors=True)
    return g


if __name__ == "__main__":
    rd = Path(sys.argv[1])
    res = grade(rd)
    write_json(rd / "grade.json", res)
    print(json.dumps({k: res.get(k) for k in ("auto_score", "points", "smoke_regrade", "rebuild", "env_py", "assets", "visual_check")}, indent=1, default=str)[:4000])
