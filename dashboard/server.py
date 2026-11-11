"""SimForge Studio: local dashboard for the sim-env-builder skill.

    .venv/bin/python -m uvicorn dashboard.server:app --port 8766        (from the repo root)
    open http://127.0.0.1:8766

Views: Studio (describe an environment -> watch a fresh agent confirm, fetch assets, build,
smoke-test; answer its questions), Golden set (scores per iteration/task, judge findings),
Environments (gallery) and a live 3D viewer that runs the MuJoCo scene on the server and streams
poses to three.js over a WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(HERE))

from common import GOLDEN, RUNS, STUDIO_RUNS, find_env_dir, read_json, run_agent, set_status, write_json  # noqa: E402
from events import parse_transcript  # noqa: E402

app = FastAPI(title="SimForge Studio")
STUDIO_RUNS.mkdir(parents=True, exist_ok=True)
_running: dict[str, threading.Thread] = {}
_scene_cache: dict[str, tuple[float, dict]] = {}
_events_cache: dict[str, tuple[float, dict]] = {}

# ----------------------------------------------------------------------------- run ids


def run_dir(rid: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(--[A-Za-z0-9_.-]+){1,2}", rid):
        raise HTTPException(400, "bad run id")
    parts = rid.split("--")
    if parts[0] == "golden" and len(parts) == 3:
        d = RUNS / parts[1] / parts[2]
    elif parts[0] == "studio" and len(parts) == 2:
        d = STUDIO_RUNS / parts[1]
    else:
        raise HTTPException(404, "unknown run")
    if not d.exists():
        raise HTTPException(404, "run not found")
    return d


def env_of(d: Path) -> Path | None:
    meta = read_json(d / "run_meta.json", {}) or {}
    if meta.get("env_dir") and Path(meta["env_dir"]).exists():
        return Path(meta["env_dir"])
    return find_env_dir(d / "work") if (d / "work").exists() else None


def _thumb(d: Path, env: Path | None) -> str | None:
    for cand in ([env / "smoke/renders"] if env else []) + [d / "regrade/renders"]:
        if cand.exists():
            for pref in ("cam_", "auto_overview", "auto_"):
                hits = sorted(cand.glob(f"{pref}*.png"))
                if hits:
                    return str(hits[0].relative_to(d))
    return None


def summarize_run(rid: str, d: Path) -> dict:
    st = read_json(d / "status.json", {}) or {}
    task = read_json(d / "task.json", {}) or {}
    meta = read_json(d / "meta.json", {}) or {}
    res = read_json(d / "result.json", {}) or {}
    env = env_of(d)
    smoke = read_json(env / "smoke/smoke_report.json", {}) if env else {}
    return {"id": rid, "kind": rid.split("--")[0], "title": meta.get("title") or task.get("id") or d.name,
            "category": task.get("category") or meta.get("category"), "prompt": task.get("prompt") or meta.get("prompt"),
            "phase": st.get("phase"), "updated": st.get("updated") or d.stat().st_mtime, "score": res.get("total_score"),
            "verdict": (smoke or {}).get("verdict"), "has_env": bool(env and (env / "scene.xml").exists()),
            "thumb": _thumb(d, env), "iteration": rid.split("--")[1] if rid.startswith("golden") else None}


def all_runs() -> list[dict]:
    out = []
    for it in sorted(RUNS.glob("iter_*")):
        for d in sorted(p for p in it.iterdir() if p.is_dir() and (p / "task.json").exists()):
            out.append(summarize_run(f"golden--{it.name}--{d.name}", d))
    for d in sorted(p for p in STUDIO_RUNS.iterdir() if p.is_dir()):
        out.append(summarize_run(f"studio--{d.name}", d))
    out.sort(key=lambda r: -(r["updated"] or 0))
    return out


# ----------------------------------------------------------------------------- read APIs


@app.get("/api/runs")
def api_runs():
    return all_runs()


@app.get("/api/golden")
def api_golden():
    from ledger import build_ledger

    board = build_ledger()
    tasks = json.loads((GOLDEN / "tasks.json").read_text())["tasks"]
    return {"board": board, "tasks": tasks}


@app.get("/api/runs/{rid}")
def api_run(rid: str):
    d = run_dir(rid)
    env = env_of(d)
    info = summarize_run(rid, d)
    info.update(status=read_json(d / "status.json", {}), task=read_json(d / "task.json"), meta=read_json(d / "meta.json"),
                run_meta=read_json(d / "run_meta.json"), result=read_json(d / "result.json"), grade=read_json(d / "grade.json"),
                judge=read_json(d / "judge.json"), running=rid in _running and _running[rid].is_alive())
    if env:
        rel = lambda p: str(p.relative_to(d))  # noqa: E731
        docs = {}
        for name in ("SPEC.md", "README.md", "assets/ASSETS.md", "smoke/visual_check.md", "smoke/SMOKE.md", "assets/polypizza/CREDITS.md"):
            if (env / name).exists():
                docs[name] = (env / name).read_text(errors="replace")[:60000]
        renders = sorted(rel(p) for p in (env / "smoke/renders").glob("*.png")) if (env / "smoke/renders").exists() else []
        info.update(env={"dir": rel(env), "name": env.name, "docs": docs, "renders": renders,
                         "contact_sheet": rel(env / "smoke/contact_sheet.png") if (env / "smoke/contact_sheet.png").exists() else None,
                         "filmstrip": rel(env / "smoke/filmstrip.png") if (env / "smoke/filmstrip.png").exists() else None,
                         "collision_sheet": rel(env / "smoke/collision_sheet.png") if (env / "smoke/collision_sheet.png").exists() else None,
                         "video": rel(env / "smoke/rollout.mp4") if (env / "smoke/rollout.mp4").exists() else None,
                         "showcase": next((rel(p) for p in (env / "smoke/showcase.mp4", d / "regrade/showcase.mp4", d / "showcase.mp4") if p.exists()), None),
                         "physics": read_json(env / "smoke/physics_report.json"), "render": read_json(env / "smoke/render_report.json"),
                         "smoke": read_json(env / "smoke/smoke_report.json"), "manifest": read_json(env / "smoke/env_manifest.json"),
                         "polypizza": read_json(env / "assets/polypizza/polypizza_assets.json")})
    return info


@app.get("/api/runs/{rid}/events")
def api_events(rid: str, since: int = 0):
    d = run_dir(rid)
    tr = d / "transcript.jsonl"
    mt = tr.stat().st_mtime if tr.exists() else 0
    hit = _events_cache.get(rid)
    if not hit or hit[0] != mt:
        hit = (mt, parse_transcript(d, 0))
        _events_cache[rid] = hit
    full = hit[1]
    st = read_json(d / "status.json", {}) or {}
    return {"events": [e for e in full["events"] if e["i"] >= since], "next": full["next"], "stage": full["stage"],
            "cost": full["cost"], "status": st, "running": rid in _running and _running[rid].is_alive()}


@app.get("/api/runs/{rid}/file")
def api_file(rid: str, path: str):
    d = run_dir(rid).resolve()
    p = (d / path).resolve() if not path.startswith("/") else Path(path).resolve()
    if d not in p.parents and p != d:
        raise HTTPException(403, "outside run dir")
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p)


# ----------------------------------------------------------------------------- studio


def _studio_shot(rid: str, d: Path, text: str, resume: bool):
    st = read_json(d / "status.json", {}) or {}
    sid = st.get("session_id") or str(uuid.uuid4())
    shot = int(st.get("shot", 0)) + 1
    set_status(d, phase="running", shot=shot, session_id=sid)
    try:
        r = run_agent(d / "work", text, sid, resume=resume, shot=shot, transcript=d / "transcript.jsonl")
        (d / f"shot{shot}_reply.md").write_text(r.get("result") or "")
        env = find_env_dir(d / "work")
        built = bool(env and (env / "smoke/smoke_report.json").exists())
        if env:
            meta = read_json(d / "run_meta.json", {}) or {}
            meta.update(env_dir=str(env), session_id=sid, shots_used=shot)
            write_json(d / "run_meta.json", meta)
        set_status(d, phase="built" if built else "awaiting_reply", last_reply=(r.get("result") or "")[:20000],
                   cost=(st.get("cost") or 0) + (r.get("cost_usd") or 0), error=r.get("is_error"))
    except Exception as e:  # noqa: BLE001
        set_status(d, phase="error", error=repr(e))


def _start(rid: str, target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    _running[rid] = t
    t.start()


@app.post("/api/studio")
def api_studio_new(payload: dict = Body(...)):
    prompt = (payload.get("prompt") or "").strip()
    if len(prompt) < 8:
        raise HTTPException(400, "describe the environment in a sentence or more")
    sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = STUDIO_RUNS / sid
    (d / "work").mkdir(parents=True)
    title = payload.get("title") or re.sub(r"\s+", " ", prompt)[:60]
    write_json(d / "meta.json", {"title": title, "prompt": prompt, "created": time.time(), "category": payload.get("category")})
    set_status(d, phase="queued", shot=0, session_id=str(uuid.uuid4()), started=time.time())
    rid = f"studio--{sid}"
    _start(rid, _studio_shot, rid, d, prompt, False)
    return {"id": rid}


@app.post("/api/runs/{rid}/reply")
def api_reply(rid: str, payload: dict = Body(...)):
    d = run_dir(rid)
    if not rid.startswith("studio--"):
        raise HTTPException(400, "golden runs are read-only")
    if rid in _running and _running[rid].is_alive():
        raise HTTPException(409, "agent is still working")
    text = (payload.get("text") or "").strip() or "Go ahead with your recommended defaults."
    _start(rid, _studio_shot, rid, d, text, True)
    return {"ok": True}


def _evaluate(rid: str, d: Path):
    try:
        set_status(d, phase="grading")
        from grade import grade

        g = grade(d)
        write_json(d / "grade.json", g)
        set_status(d, phase="judging")
        meta = read_json(d / "meta.json", {}) or {}
        write_json(d / "task.json", {"id": d.name, "category": meta.get("category") or "studio", "prompt": meta.get("prompt"),
                                     "user_reply": "(interactive)", "expected_backend": "agent's choice", "must_have": [], "checks": []})
        from judge import run_judge

        j = run_judge(d) or {}
        write_json(d / "result.json", {"auto_score": g.get("auto_score"), "judge_score": j.get("judge_score"),
                                       "total_score": round((g.get("auto_score") or 0) + (j.get("judge_score") or 0), 1)})
        set_status(d, phase="built")
    except Exception as e:  # noqa: BLE001
        set_status(d, phase="error", error=repr(e))


@app.post("/api/runs/{rid}/evaluate")
def api_evaluate(rid: str):
    d = run_dir(rid)
    if rid in _running and _running[rid].is_alive():
        raise HTTPException(409, "busy")
    _start(rid, _evaluate, rid, d)
    return {"ok": True}


# ----------------------------------------------------------------------------- simulation


@app.get("/api/runs/{rid}/scene")
def api_scene(rid: str):
    d = run_dir(rid)
    env = env_of(d)
    if not env or not (env / "scene.xml").exists():
        raise HTTPException(404, "no scene.xml yet")
    import mujoco
    from sim_session import export_scene

    mt = (env / "scene.xml").stat().st_mtime
    hit = _scene_cache.get(rid)
    if hit and hit[0] == mt:
        return JSONResponse(hit[1])
    try:
        model = mujoco.MjModel.from_xml_path(str(env / "scene.xml"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"scene does not compile: {e}") from e
    sc = export_scene(model)
    sc["has_controller"] = (env / "controller.py").exists()
    _scene_cache[rid] = (mt, sc)
    return JSONResponse(sc)


_sessions: dict[str, object] = {}


@app.get("/api/runs/{rid}/render")
def api_render(rid: str, camera: str = "-1"):
    from sim_session import SimSession

    d = run_dir(rid)
    env = env_of(d)
    if not env:
        raise HTTPException(404, "no env")
    s = _sessions.get(rid) or SimSession(env)
    cam = int(camera) if camera.lstrip("-").isdigit() else camera
    return Response(s.render_png(cam), media_type="image/png")


@app.websocket("/api/runs/{rid}/sim")
async def ws_sim(ws: WebSocket, rid: str):
    from sim_session import SimSession

    await ws.accept()
    try:
        d = run_dir(rid)
        env = env_of(d)
        sess = await asyncio.to_thread(SimSession, env)
    except Exception as e:  # noqa: BLE001
        await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        await ws.close()
        return
    old = _sessions.pop(rid, None)
    if old is not None:
        old.close()
    _sessions[rid] = sess
    await ws.send_text(json.dumps({"type": "hello", **sess.info()}))
    stop = asyncio.Event()

    async def pump():
        last_info = 0.0
        while not stop.is_set():
            await ws.send_bytes(sess.frame())
            if time.time() - last_info > 0.5:
                await ws.send_text(json.dumps({"type": "info", **sess.info()}))
                last_info = time.time()
            await asyncio.sleep(1 / 40)

    task = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("cmd") == "push" and msg.get("body"):
                b = int(msg["body"])
                m = float(sess.model.body_subtreemass[b]) if 0 < b < sess.model.nbody else 0.0
                f = [float(x) * m * 25.0 for x in msg.get("dir", [0, 0, 1])]
                sess.command({"cmd": "push", "body": b, "force": f})
            else:
                sess.command(msg)
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        stop.set()
        task.cancel()
        sess.close()
        if _sessions.get(rid) is sess:
            _sessions.pop(rid, None)


# ----------------------------------------------------------------------------- frontend

DIST = HERE / "web/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    idx = DIST / "index.html"
    if not idx.exists():
        return Response("frontend not built: cd dashboard/web && npm install && npm run build", media_type="text/plain")
    f = (DIST / path).resolve()
    if path and DIST in f.parents and f.is_file():
        return FileResponse(f)
    return FileResponse(idx)
