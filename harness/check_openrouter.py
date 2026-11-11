"""Check that headless agents can run through OpenRouter (tiny, cheap request). Never prints the key.

    .venv/bin/python harness/check_openrouter.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLAUDE, OPENROUTER_KEY_FILE, agent_env, openrouter_key, provider  # noqa: E402

key = openrouter_key()
if not key:
    print(f"no key found: set OPENROUTER_API_KEY or create {OPENROUTER_KEY_FILE}")
    sys.exit(1)
mode = OPENROUTER_KEY_FILE.stat().st_mode & 0o777 if OPENROUTER_KEY_FILE.exists() else None
print(f"key found ({'env var' if mode is None else f'file, permissions {oct(mode)}'}), starts with {key[:6]}..., provider = {provider()}")
env, model = agent_env()
cmd = [CLAUDE, "-p", "Reply with exactly: OPENROUTER_OK", "--strict-mcp-config", "--setting-sources", "project",
       "--no-chrome", "--tools", "", "--model", model, "--output-format", "json"]
with tempfile.TemporaryDirectory() as d:
    p = subprocess.run(cmd, cwd=d, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
try:
    r = json.loads(p.stdout)
    print("model:", model, "| result:", r.get("result"), "| error:", r.get("is_error"), "| cost:", r.get("total_cost_usd"))
    sys.exit(0 if "OPENROUTER_OK" in (r.get("result") or "") else 2)
except ValueError:
    print("unexpected output:", (p.stdout + p.stderr)[-800:].replace(key, "<key>"))
    sys.exit(2)
