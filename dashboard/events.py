"""Turn a Claude stream-json transcript into a compact, UI-friendly timeline.

Each UI event: {"i": n, "t": unix, "shot": k, "kind": ..., ...}
kinds: shot_start, text, tool, tool_result, shot_end, image
Also infers the build stage (confirm, assets, build, smoke, visual, deliver) from tool use.
Images inside tool results (e.g. Poly Pizza preview grids) are written to <run>/_media/ and referenced.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

STAGES = ["confirm", "assets", "build", "smoke", "visual", "deliver"]


def _short(s, n=220):
    s = str(s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _tool_summary(name: str, inp: dict) -> tuple[str, str]:
    """(label, detail) for a tool call."""
    n = name.split("__")[-1] if name.startswith("mcp__") else name
    if name.startswith("mcp__polypizza"):
        if n == "find_assets":
            roles = [x.get("role") for x in inp.get("needs", [])]
            return "Poly Pizza · find assets", ", ".join(map(str, roles))
        if n == "search_models":
            return "Poly Pizza · search", inp.get("query", "")
        if n == "preview_models":
            return "Poly Pizza · preview", f"{len(inp.get('model_ids', []))} models"
        if n == "download_model":
            return "Poly Pizza · download", f"{inp.get('name') or ''} ({inp.get('model_id')})"
        return f"Poly Pizza · {n}", _short(json.dumps(inp), 120)
    if n == "Bash":
        return "Shell", _short(inp.get("command"), 260)
    if n in ("Read", "Write", "Edit"):
        return n, inp.get("file_path", "")
    if n == "Skill":
        return "Skill", inp.get("skill", "")
    if n in ("Glob", "Grep"):
        return n, inp.get("pattern", "")
    if n == "TodoWrite":
        return "Plan", f"{len(inp.get('todos', []))} items"
    return n, _short(json.dumps(inp), 160)


def stage_of(name: str, inp: dict, current: str) -> str:
    cmd = str(inp.get("command", "")) if isinstance(inp, dict) else ""
    fp = str(inp.get("file_path", "")) if isinstance(inp, dict) else ""
    if name.startswith("mcp__polypizza"):
        return max(current, "assets", key=STAGES.index)
    if re.search(r"-m\s+simkit\.(smoke|physics_exam)\b", cmd):
        return max(current, "smoke", key=STAGES.index)
    if name == "Read" and re.search(r"(contact_sheet|filmstrip|collision_sheet|renders/).*\.png$", fp):
        return max(current, "visual", key=STAGES.index)
    if fp.endswith("README.md") and name in ("Write", "Edit") and STAGES.index(current) >= STAGES.index("smoke"):
        return max(current, "deliver", key=STAGES.index)
    if (name in ("Write", "Edit") and fp.endswith((".py", ".xml"))) or "build_env.py" in cmd:
        return max(current, "build", key=STAGES.index)
    return current


def parse_transcript(run_dir: Path, since: int = 0) -> dict:
    path = Path(run_dir) / "transcript.jsonl"
    media = Path(run_dir) / "_media"
    events, stage, cost, pending = [], "confirm", 0.0, {}
    if not path.exists():
        return {"events": [], "stage": stage, "next": 0, "cost": 0.0}
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        typ = ev.get("type")
        shot = ev.get("_shot") or ev.get("shot")
        t = ev.get("_t") or ev.get("t")
        out = []
        if typ == "harness" and ev.get("event") == "shot_start":
            out.append({"kind": "shot_start", "prompt": ev.get("prompt", "")})
        elif typ == "harness" and ev.get("event") == "shot_end":
            out.append({"kind": "shot_end", "cost": ev.get("cost_usd"), "turns": ev.get("num_turns"), "error": ev.get("is_error"),
                        "timed_out": ev.get("timed_out")})
        elif typ == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text", "").strip():
                    out.append({"kind": "text", "text": c["text"]})
                elif c.get("type") == "tool_use":
                    name, inp = c.get("name", ""), c.get("input") or {}
                    label, detail = _tool_summary(name, inp)
                    stage = stage_of(name, inp, stage)
                    e = {"kind": "tool", "id": c.get("id"), "name": name, "label": label, "detail": detail, "stage": stage}
                    if name == "Read" and str(inp.get("file_path", "")).lower().endswith((".png", ".jpg", ".webp")):
                        e["image_path"] = inp.get("file_path")
                    pending[c.get("id")] = name
                    out.append(e)
        elif typ == "user":
            for c in (ev.get("message") or {}).get("content") or []:
                if not isinstance(c, dict) or c.get("type") != "tool_result":
                    continue
                tid = c.get("tool_use_id")
                content = c.get("content")
                texts, imgs = [], []
                if isinstance(content, list):
                    for k, part in enumerate(content):
                        if part.get("type") == "text":
                            texts.append(part.get("text", ""))
                        elif part.get("type") == "image":
                            src = part.get("source") or {}
                            if src.get("type") == "base64" and pending.get(tid, "").startswith("mcp__polypizza"):
                                media.mkdir(exist_ok=True)
                                fn = media / f"{tid}_{k}.png"
                                if not fn.exists():
                                    fn.write_bytes(base64.b64decode(src.get("data", "")))
                                imgs.append(f"_media/{fn.name}")
                else:
                    texts.append(str(content or ""))
                txt = "\n".join(texts)
                e = {"kind": "tool_result", "id": tid, "ok": not c.get("is_error"), "name": pending.get(tid, ""), "text": _short(txt, 600)}
                if imgs:
                    e["images"] = imgs
                if pending.get(tid, "").startswith("mcp__polypizza"):
                    try:
                        js = json.loads(txt) if txt.strip().startswith("{") else None
                        if isinstance(js, dict) and set(js) == {"result"} and isinstance(js["result"], str):
                            js = json.loads(js["result"])  # MCP SDK structured-output wrapper
                    except ValueError:
                        js = None
                    if js:
                        e["polypizza"] = _pp_digest(pending[tid], js)
                out.append(e)
        elif typ == "result":
            cost += ev.get("total_cost_usd") or 0
        for o in out:
            o.update(i=i, t=t, shot=shot)
            i += 1
            if o["i"] >= since:
                events.append(o)
    return {"events": events, "stage": stage, "next": i, "cost": round(cost, 3)}


def _pp_digest(tool: str, js: dict) -> dict:
    n = tool.split("__")[-1]
    if n == "find_assets":
        return {"type": "find", "roles": {r: {"verdict": v.get("verdict"), "top": [c.get("title") for c in v.get("candidates", [])[:3]]}
                                          for r, v in (js.get("roles") or {}).items()}}
    if n == "search_models":
        return {"type": "search", "query": js.get("query"), "verdict": js.get("verdict"), "top": [r.get("title") for r in js.get("results", [])[:4]]}
    if n == "download_model":
        m = js.get("model") or {}
        return {"type": "download", "ok": js.get("ok"), "title": m.get("title"), "licence": m.get("licence"), "creator": m.get("creator"),
                "preview": m.get("preview_url"), "size": (js.get("geometry") or {}).get("size_zup_xyz")}
    return {"type": n}
