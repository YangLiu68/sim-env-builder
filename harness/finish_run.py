"""Finish a golden run whose harness stopped after building: grade (if missing), judge (if missing),
write result.json, mark done and refresh the ledger.

    .venv/bin/python harness/finish_run.py golden/runs/iter_02/quadruped_trail [--regrade] [--rejudge]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, set_status, write_json  # noqa: E402
from grade import grade  # noqa: E402
from judge import run_judge  # noqa: E402
from ledger import build_ledger  # noqa: E402

rd = Path(sys.argv[1]).resolve()
meta = read_json(rd / "run_meta.json", {}) or {}
g = read_json(rd / "grade.json")
if g is None or "--regrade" in sys.argv:
    set_status(rd, phase="grading")
    g = grade(rd)
    write_json(rd / "grade.json", g)
j = read_json(rd / "judge.json")
if not (isinstance(j, dict) and "judge_score" in j) or "--rejudge" in sys.argv:
    set_status(rd, phase="judging")
    j = run_judge(rd) or {}
result = {**meta, "auto_score": g.get("auto_score"), "judge_score": j.get("judge_score"),
          "total_score": round((g.get("auto_score") or 0) + (j.get("judge_score") or 0), 1),
          "smoke_regrade": (g.get("smoke_regrade") or {}).get("verdict"), "finished": time.time()}
write_json(rd / "result.json", result)
set_status(rd, phase="done", total_score=result["total_score"])
build_ledger()
print(f"{rd.name}: total {result['total_score']} (auto {result['auto_score']}, judge {result['judge_score']})")
