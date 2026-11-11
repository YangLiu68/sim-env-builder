"""Aggregate all golden runs into golden/scoreboard.json and golden/LEDGER.md."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GOLDEN, RUNS, read_json, write_json  # noqa: E402


def build_ledger() -> dict:
    tasks = {t["id"]: t for t in json.loads((GOLDEN / "tasks.json").read_text())["tasks"]}
    iters = []
    for it in sorted(RUNS.glob("iter_*")):
        info = read_json(it / "iteration.json", {}) or {}
        rows = []
        for rd in sorted(p for p in it.iterdir() if p.is_dir() and (p / "task.json").exists()):
            r = read_json(rd / "result.json") or {}
            st = read_json(rd / "status.json") or {}
            j = read_json(rd / "judge.json") or {}
            g = read_json(rd / "grade.json") or {}
            rows.append({"task": rd.name, "category": tasks.get(rd.name, {}).get("category"), "phase": st.get("phase"),
                         "total": r.get("total_score"), "auto": r.get("auto_score"), "judge": r.get("judge_score"),
                         "shots": r.get("shots_used"), "cost": r.get("agent_cost_usd"), "minutes": round((r.get("build_seconds") or 0) / 60, 1),
                         "smoke": r.get("smoke_regrade"), "scores": j.get("scores"), "missing": j.get("must_have_missing"),
                         "problems": j.get("problems", []), "summary": j.get("summary"), "points": g.get("points")})
        done = [x for x in rows if x["total"] is not None]
        void = str(info.get("notes", "")).upper().startswith("VOID")  # infrastructure failure, not a skill result
        iters.append({"iter": it.name, "notes": info.get("notes", ""), "rows": rows, "void": void,
                      "mean_total": None if void else (round(sum(x["total"] for x in done) / len(done), 1) if done else None),
                      "n_done": len(done), "n": len(rows)})
    board = {"updated": time.strftime("%Y-%m-%d %H:%M"), "iterations": iters}
    write_json(GOLDEN / "scoreboard.json", board)
    lines = ["# Golden set ledger", "", f"Updated {board['updated']}. Score = auto (0-40) + judge (0-60).", ""]
    lines += ["| iteration | tasks done | mean score | notes |", "|---|---|---|---|"]
    for it in iters:
        lines.append(f"| {it['iter']} | {it['n_done']}/{it['n']} | {it['mean_total']} | {it['notes'][:120]} |")
    for it in iters:
        lines += ["", f"## {it['iter']}", "", "| task | total | auto | judge | shots | smoke | min | $ | top problem |", "|---|---|---|---|---|---|---|---|---|"]
        for x in it["rows"]:
            top = next((p for p in x["problems"] if p.get("severity") == "high"), (x["problems"] or [{}])[0] if x["problems"] else {})
            lines.append(f"| {x['task']} | {x['total']} | {x['auto']} | {x['judge']} | {x['shots']} | {x['smoke']} | {x['minutes']} | {x['cost']} | {str(top.get('description', ''))[:110]} |")
    (GOLDEN / "LEDGER.md").write_text("\n".join(lines) + "\n")
    return board


if __name__ == "__main__":
    b = build_ledger()
    print(json.dumps([{k: v for k, v in it.items() if k != "rows"} for it in b["iterations"]], indent=1))
