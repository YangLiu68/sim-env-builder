"""Resume a golden run whose agent was cut off by an outage or usage limit (same session,
"continue" message), then grade + judge it. Outage resumes are recorded, not counted as user shots.

    .venv/bin/python harness/resume_run.py golden/runs/iter_03/harbor_boat
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTINUE_AFTER_OUTAGE, find_env_dir, read_json, run_agent, set_status, write_json  # noqa: E402

rd = Path(sys.argv[1]).resolve()
st = read_json(rd / "status.json", {}) or {}
sid = st.get("session_id")
if not sid:
    sys.exit("no session_id in status.json")
work = rd / "work"
plugin = rd.parent / "plugin_snapshot"
shot = int(str(st.get("phase", "shot2")).replace("shot", "") or 2) if str(st.get("phase", "")).startswith("shot") else 2
user_shots = min(shot, 2) if (rd / "shot1_reply.md").exists() else 1
t0 = st.get("started") or time.time()
resumes = 0
env = find_env_dir(work)
while resumes < 3 and not (env and (env / "smoke/smoke_report.json").exists()):
    resumes += 1
    set_status(rd, phase=f"resume{resumes}")
    r = run_agent(work, CONTINUE_AFTER_OUTAGE, sid, resume=True, shot=shot, transcript=rd / "transcript.jsonl", plugin=plugin)
    (rd / f"resume{resumes}_reply.md").write_text(r.get("result") or "")
    env = find_env_dir(work)
    if r.get("is_error") and not r.get("transient"):
        break
meta = read_json(rd / "run_meta.json", {}) or {}
meta.update(task=rd.name, session_id=sid, shots_used=user_shots, outage_resumes=resumes, env_dir=str(env) if env else None,
            built_before_confirm=meta.get("built_before_confirm", False),
            asked_confirmation=meta.get("asked_confirmation", "?" in ((rd / "shot1_reply.md").read_text() if (rd / "shot1_reply.md").exists() else "")),
            build_seconds=round(time.time() - t0))
write_json(rd / "run_meta.json", meta)
import subprocess  # noqa: E402

subprocess.run([sys.executable, str(Path(__file__).with_name("finish_run.py")), str(rd), "--regrade"], check=False)
