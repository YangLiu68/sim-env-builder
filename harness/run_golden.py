"""Run golden-set tasks end to end with FRESH headless Claude agents (zero prior context).

Each agent gets only: the sim-env-builder plugin (skill + playbook + scripts), the Poly Pizza
MCP server, and the task prompt. Flow per task:

  shot 1  user prompt              -> agent should reply with a confirmation brief (no build yet)
  shot 2  simulated user's answers -> agent builds + smoke-tests the environment
  shot 3  (only if unfinished)     -> "please finish"  (counts against the few-shot goal)
  grade   harness/grade.py         -> automated checks, independent smoke re-run
  judge   fresh claude -p          -> rubric scores after looking at the renders

    python harness/run_golden.py --iter 1 --tier core --parallel 3
    python harness/run_golden.py --iter 2 --tasks drone_gates,arm_kitchen
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GOLDEN, RUNS, find_env_dir, read_json, run_agent, set_status, write_json  # noqa: E402

TIMEOUT_PROMPT = ("Time is up. Stop tuning now: run the smoke test once more if your last change is untested, "
                  "make REPORT.md and the README results final (list anything dropped or unverified), and send "
                  "me the final report.")
FINISH_PROMPT = ("Please continue and finish everything from the playbook: build, smoke tests (render, "
                 "physics examiner, your own visual check of the images), fixes, README. Then give me the final summary.")


def load_tasks():
    return {t["id"]: t for t in json.loads((GOLDEN / "tasks.json").read_text())["tasks"]}


def smoke_done(env_dir: Path | None) -> bool:
    return bool(env_dir and (env_dir / "smoke/smoke_report.json").exists())


def run_task(task: dict, iter_dir: Path, max_shots: int = 3, judge: bool = True) -> dict:
    run_dir = iter_dir / task["id"]
    plugin = iter_dir / "plugin_snapshot"  # frozen copy: edits to plugin/ during the iteration cannot leak in
    work = run_dir / "work"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    work.mkdir(parents=True)
    write_json(run_dir / "task.json", task)
    sid = str(uuid.uuid4())
    t0 = time.time()
    shots, total_cost = [], 0.0
    set_status(run_dir, phase="shot1", task=task["id"], session_id=sid, started=t0)
    r1 = run_agent(work, task["prompt"], sid, resume=False, shot=1, transcript=run_dir / "transcript.jsonl", plugin=plugin)
    shots.append({"shot": 1, **{k: v for k, v in r1.items() if k != "result"}})
    (run_dir / "shot1_reply.md").write_text(r1["result"] or "")
    total_cost = max(total_cost, r1["cost_usd"] or 0)  # session-cumulative
    env_dir = find_env_dir(work)
    built_before_confirm = env_dir is not None
    asked = (not built_before_confirm) and ("?" in (r1["result"] or ""))
    n = 1
    if not built_before_confirm and not r1["timed_out"]:
        n = 2
        set_status(run_dir, phase="shot2")
        r2 = run_agent(work, task["user_reply"], sid, resume=True, shot=2, transcript=run_dir / "transcript.jsonl", plugin=plugin)
        shots.append({"shot": 2, **{k: v for k, v in r2.items() if k != "result"}})
        (run_dir / "shot2_reply.md").write_text(r2["result"] or "")
        total_cost = max(total_cost, r2["cost_usd"] or 0)  # session-cumulative
    env_dir = find_env_dir(work)
    last = shots[-1] if shots else {}
    if last.get("timed_out") and smoke_done(env_dir) and n < max_shots + 1:
        n += 1  # counts against the few-shot goal: the build overran its time
        set_status(run_dir, phase=f"shot{n}")
        rn = run_agent(work, TIMEOUT_PROMPT, sid, resume=True, shot=n, transcript=run_dir / "transcript.jsonl", plugin=plugin)
        shots.append({"shot": n, **{k: v for k, v in rn.items() if k != "result"}})
        (run_dir / f"shot{n}_reply.md").write_text(rn["result"] or "")
        total_cost = max(total_cost, rn["cost_usd"] or 0)
    while n < max_shots and not smoke_done(env_dir):
        n += 1
        set_status(run_dir, phase=f"shot{n}")
        rn = run_agent(work, FINISH_PROMPT, sid, resume=True, shot=n, transcript=run_dir / "transcript.jsonl", plugin=plugin)
        shots.append({"shot": n, **{k: v for k, v in rn.items() if k != "result"}})
        (run_dir / f"shot{n}_reply.md").write_text(rn["result"] or "")
        total_cost = max(total_cost, rn["cost_usd"] or 0)  # session-cumulative
        env_dir = find_env_dir(work)
    build_s = time.time() - t0
    meta = {"task": task["id"], "session_id": sid, "shots": shots, "shots_used": n, "built_before_confirm": built_before_confirm,
            "asked_confirmation": asked, "env_dir": str(env_dir) if env_dir else None, "build_seconds": round(build_s),
            "agent_cost_usd": round(total_cost, 2)}
    write_json(run_dir / "run_meta.json", meta)
    set_status(run_dir, phase="grading")
    from grade import grade  # local import: grade imports mujoco lazily via subprocess only

    g = grade(run_dir)
    write_json(run_dir / "grade.json", g)
    j = None
    if judge:
        set_status(run_dir, phase="judging")
        from judge import run_judge

        j = run_judge(run_dir)
    result = {**meta, "auto_score": g.get("auto_score"), "judge_score": (j or {}).get("judge_score"),
              "total_score": round((g.get("auto_score") or 0) + ((j or {}).get("judge_score") or 0), 1),
              "smoke_regrade": g.get("smoke_regrade", {}).get("verdict"), "finished": time.time()}
    write_json(run_dir / "result.json", result)
    set_status(run_dir, phase="done", total_score=result["total_score"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tier", default="")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    tasks = load_tasks()
    ids = [t for t in a.tasks.split(",") if t] or [k for k, v in tasks.items() if not a.tier or v["tier"] == a.tier]
    iter_dir = RUNS / f"iter_{a.iter:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    plugin_snapshot = iter_dir / "plugin_snapshot"
    if not plugin_snapshot.exists():  # what the agents got this iteration (for diffs between iterations)
        shutil.copytree(GOLDEN.parent / "plugin", plugin_snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(iter_dir / "iteration.json", {"iter": a.iter, "tasks": ids, "started": time.time(),
                                            "notes": read_json(iter_dir / "iteration.json", {}).get("notes", "")})
    print(f"iteration {a.iter}: {ids}", flush=True)
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        futs = {ex.submit(run_task, tasks[i], iter_dir, 3, not a.no_judge): i for i in ids}
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                r = f.result()
                print(f"[done] {i}: total {r['total_score']} (auto {r['auto_score']}, judge {r['judge_score']}), shots {r['shots_used']}, ${r['agent_cost_usd']}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {i}: {e!r}", flush=True)
    from ledger import build_ledger

    build_ledger()


if __name__ == "__main__":
    main()
