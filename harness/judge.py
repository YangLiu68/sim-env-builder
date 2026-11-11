"""Independent LLM judge for a golden run (fresh headless Claude, no plugin, read-only tools + Write).

    python harness/judge.py golden/runs/iter_01/drone_gates
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLAUDE, GOLDEN, read_json, write_json  # noqa: E402

try:
    from common import agent_env  # noqa: E402
except ImportError:  # a long-running harness may hold an older common module in memory
    from common import MODEL as _MODEL  # noqa: E402

    def agent_env():
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        return env, _MODEL

WEIGHTS = {"alignment": 1.0, "completeness": 1.5, "assets": 1.0, "visual": 1.5, "physics": 1.0}


def run_judge(run_dir: Path, timeout_s: int = 1800) -> dict | None:
    run_dir = Path(run_dir).resolve()
    task = read_json(run_dir / "task.json", {})
    meta = read_json(run_dir / "run_meta.json", {})
    env_dir = meta.get("env_dir") or "work"
    prompt = (GOLDEN / "judge_prompt.md").read_text().format(
        run_dir=run_dir, category=task.get("category"), prompt=task.get("prompt"), user_reply=task.get("user_reply"),
        expected_backend=task.get("expected_backend"), env_dir=env_dir,
        must_have="\n".join(f"- {m}" for m in task.get("must_have", [])),
        checks="\n".join(f"- {c}" for c in task.get("checks", [])))
    jf = run_dir / "judge.json"
    if jf.exists():
        jf.unlink()
    env, model = agent_env()
    cmd = [CLAUDE, "-p", prompt, "--strict-mcp-config", "--setting-sources", "project", "--no-chrome",
           "--dangerously-skip-permissions", "--tools", "Read,Glob,Grep,Write", "--model", model,
           "--output-format", "stream-json", "--verbose", "--session-id", str(uuid.uuid4())]
    t0 = time.time()
    for attempt in range(3):  # a usage limit or outage must not lose the judgement: wait and retry
        tr = run_dir / "judge_transcript.jsonl"
        with open(tr, "w") as log:
            try:
                subprocess.run(cmd[:-1] + [str(uuid.uuid4())], cwd=str(run_dir), stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, timeout=timeout_s, env=env)
            except subprocess.TimeoutExpired:
                pass
        j = read_json(jf)
        if isinstance(j, dict) and "scores" in j:
            break
        text = tr.read_text(errors="replace")[-4000:]
        if "usage_limit_reached" in text or "hit your" in text:
            import re

            from common import wait_for_usage_reset

            m = re.search(r"You've hit your[^\"]*", text)
            wait_for_usage_reset(m.group(0) if m else "")
        elif "connection_failed" in text or "Can't reach the API" in text:
            from common import wait_for_network

            wait_for_network()
        else:
            break
    j = read_json(jf)
    if not isinstance(j, dict) or "scores" not in j:
        return {"error": "judge produced no valid judge.json", "judge_score": None}
    s = j["scores"]
    tot = sum(WEIGHTS[k] * float(s.get(k, 0)) for k in WEIGHTS)
    j["judge_score"] = round(tot / (10 * sum(WEIGHTS.values())) * 60, 1)
    j["judge_seconds"] = round(time.time() - t0)
    write_json(jf, j)
    return j


if __name__ == "__main__":
    print(json.dumps(run_judge(Path(sys.argv[1])), indent=1)[:3000])
