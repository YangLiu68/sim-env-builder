"""Poly Pizza MCP server: find, preview, inspect and download free low-poly 3D models.

Run: ``python -m polypizza_mcp`` (stdio). Tools return JSON text (and one preview image tool).
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

from . import client
from .client import ModelInfo, PolyPizzaError

INSTRUCTIONS = """\
Poly Pizza (https://poly.pizza) hosts ~10k free low-poly models (CC0 or CC-BY) as GLB files.
Workflow for a simulated scene:
1. find_assets([...roles...]) or search_models(query) for every object the scene needs. Use short,
   concrete nouns ("sedan", "office chair", "traffic cone"); try 1-3 synonyms per role.
2. Read each verdict: "found" -> pick a candidate; "weak" -> look at preview_models before using;
   "not_found" -> do NOT force a bad match, generate the object procedurally in code instead.
3. preview_models(ids) shows labelled thumbnails so you can check the look before downloading.
4. download_model(id, dest_dir, name) saves <name>.glb, reports its size/orientation, and
   records the licence in dest_dir/polypizza_assets.json and dest_dir/CREDITS.md.
Models are static meshes (no joints). Sizes are in arbitrary file units: always rescale to
real-world dimensions and confirm orientation with a render. CC-BY models need the credit line.
Without an API key the server reads the public website politely (10 s between page requests,
results cached for a week), so batch your searches with find_assets.
"""

mcp = MCPServer("polypizza", instructions=INSTRUCTIONS, version="0.2.0")


def _licence_ok(info: ModelInfo, licence: str) -> bool:
    lic = (info.licence or "").upper()
    want = licence.lower().replace("_", "-")
    if want in ("any", ""):
        return True
    if want == "cc0":
        return lic.startswith("CC0")
    if want in ("cc-by", "ccby"):
        return lic.startswith("CC-BY") or lic.startswith("CC0")
    return True


def _verdict(query: str, results: list[ModelInfo]) -> tuple[str, str]:
    q = [client.match_quality(query, r) for r in results]
    if "exact" in q:
        return "found", "Relevant models exist; pick the best candidate (preview_models to compare)."
    if "partial" in q:
        return "weak", (
            "Only partial title matches: check preview_models; if none truly fits, try a synonym "
            "or generate this object procedurally."
        )
    return "not_found", "No relevant model on Poly Pizza: generate this object procedurally in code."


def _brief(r: ModelInfo, query: str, rank: int) -> dict:
    return {
        "rank": rank,
        "id": r.id,
        "title": r.title,
        "creator": r.creator,
        "licence": r.licence,
        "match": client.match_quality(query, r),
        "page_url": r.page_url,
        "glb_url": r.glb_url,
    }


@mcp.tool(structured_output=False)
def search_models(query: str, limit: int = 10, licence: str = "any") -> str:
    """Search Poly Pizza for low-poly 3D models.

    Args:
        query: short noun phrase, e.g. "forklift", "office chair", "pine tree".
        limit: max results (1-20).
        licence: "any", "cc0" (public domain only) or "cc-by" (CC0 or CC-BY).

    Returns JSON with a verdict ("found" / "weak" / "not_found"), advice, and results with
    id, title, creator, licence and match quality (exact / partial / none).
    """
    limit = max(1, min(int(limit), 20))
    try:
        res = [r for r in client.search(query, 32) if _licence_ok(r, licence)][:limit]
    except PolyPizzaError as e:
        return json.dumps({"query": query, "verdict": "error", "error": str(e),
                           "advice": "Poly Pizza unreachable: use procedural generation for now."})
    verdict, advice = _verdict(query, res)
    return json.dumps(
        {"query": query, "mode": client.mode(), "verdict": verdict, "advice": advice,
         "results": [_brief(r, query, i) for i, r in enumerate(res)]},
        indent=1,
    )


@mcp.tool(structured_output=False)
def find_assets(needs: list[dict], per_role: int = 4, licence: str = "any") -> str:
    """Batch search for every object a scene needs; one verdict per role.

    Args:
        needs: list of {"role": "car", "queries": ["sedan", "car"]}. "queries" defaults to
            [role]. Queries are tried in order; all results are merged and ranked by match.
        per_role: candidates to return per role.
        licence: "any", "cc0" or "cc-by".

    Returns JSON: {role: {verdict, advice, candidates[...]}} plus a summary listing which roles
    must be generated procedurally. Keyless mode costs ~10 s per uncached query.
    """
    out: dict[str, dict] = {}
    order = {"exact": 0, "partial": 1, "none": 2}
    for need in needs:
        role = str(need.get("role") or "").strip()
        if not role:
            continue
        queries = [q for q in (need.get("queries") or [role]) if str(q).strip()]
        seen, merged, errors = set(), [], []
        for q in queries:
            try:
                for i, r in enumerate(client.search(q, 32)):
                    if r.id in seen or not _licence_ok(r, licence):
                        continue
                    seen.add(r.id)
                    merged.append((order[client.match_quality(q, r)], len(merged), i, q, r))
            except PolyPizzaError as e:
                errors.append(f"{q}: {e}")
        merged.sort(key=lambda t: (t[0], t[2], t[1]))
        best = [m for m in merged if m[0] < 2][:per_role] or merged[:min(2, per_role)]
        quals = {client.match_quality(q, r) for _, _, _, q, r in merged}
        if "exact" in quals:
            verdict = "found"
        elif "partial" in quals:
            verdict = "weak"
        else:
            verdict = "error" if errors and not merged else "not_found"
        cands = [dict(_brief(r, q, i), query=q) for (_, _, i, q, r) in best]
        out[role] = {
            "verdict": verdict,
            "advice": {"found": "use a candidate (check preview_models)",
                       "weak": "preview first; fall back to procedural if nothing fits",
                       "not_found": "generate procedurally",
                       "error": "Poly Pizza unreachable; generate procedurally"}[verdict],
            "queries": queries,
            "candidates": cands,
            **({"errors": errors} if errors else {}),
        }
    summary = {
        "found": [k for k, v in out.items() if v["verdict"] == "found"],
        "weak": [k for k, v in out.items() if v["verdict"] == "weak"],
        "procedural_fallback_needed": [k for k, v in out.items() if v["verdict"] in ("not_found", "error")],
    }
    return json.dumps({"mode": client.mode(), "summary": summary, "roles": out}, indent=1)


def _preview_grid(infos: list[ModelInfo], cols: int = 4, tw: int = 300, th: int = 170):
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    cols = min(cols, len(infos))
    rows = (len(infos) + cols - 1) // cols
    label_h = 38
    sheet = PILImage.new("RGB", (cols * tw, rows * (th + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for i, info in enumerate(infos):
        x, y = (i % cols) * tw, (i // cols) * (th + label_h)
        try:
            raw = client.fetch_bytes_cached(info.preview_url, "preview")
            im = PILImage.open(io.BytesIO(raw)).convert("RGB")
            im.thumbnail((tw - 6, th - 6))
            sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        except Exception:  # noqa: BLE001 - missing preview: leave the tile blank
            draw.text((x + 10, y + th // 2), "no preview", fill=(120, 120, 120), font=font)
        draw.rectangle([x, y + th, x + tw - 1, y + th + label_h - 1], fill=(255, 255, 255))
        draw.text((x + 6, y + th + 3), f"[{i}] {info.id}", fill=(20, 20, 20), font=font)
        draw.text((x + 6, y + th + 20), f"{info.title[:30]} | {info.licence}", fill=(80, 80, 80), font=font)
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


@mcp.tool(structured_output=False)
def preview_models(model_ids: list[str]) -> list:
    """Labelled thumbnail grid of up to 12 models (by id) so you can judge the look.

    Use it to choose between candidates and to reject wrong matches (a toy car for "sedan",
    a cartoon for a realistic scene). Ids from earlier searches cost no extra page requests.
    """
    infos, missing = [], []
    for mid in model_ids[:12]:
        try:
            infos.append(client.lookup(mid))
        except PolyPizzaError as e:
            missing.append(f"{mid}: {e}")
    if not infos:
        return [json.dumps({"error": "no valid ids", "details": missing})]
    png = _preview_grid(infos)
    meta = [{"index": i, "id": m.id, "title": m.title, "tris": m.tris, "licence": m.licence,
             "category": m.category, "tags": m.tags} for i, m in enumerate(infos)]
    return [Image(data=png, format="png"), json.dumps({"models": meta, "missing": missing}, indent=1)]


@mcp.tool(structured_output=False)
def get_model(model_id: str) -> str:
    """Full metadata for one model: title, description, triangle count, category, tags, licence,
    creator, animated flag, GLB url and the attribution line."""
    try:
        return json.dumps(client.get_model(model_id).to_dict(), indent=1)
    except PolyPizzaError as e:
        return json.dumps({"error": str(e)})


_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _write_credits(dest: Path, manifest: dict) -> Path:
    lines = ["# 3D model credits", "",
             "Models from Poly Pizza (https://poly.pizza). CC-BY models require this attribution.", ""]
    for name, e in sorted(manifest.items()):
        lines.append(f"- `{name}`: {e['attribution']}")
    p = dest / "CREDITS.md"
    p.write_text("\n".join(lines) + "\n")
    return p


@mcp.tool(structured_output=False)
def download_model(model_id: str, dest_dir: str, name: str = "") -> str:
    """Download a model's GLB into dest_dir/<name>.glb and inspect it.

    Args:
        model_id: Poly Pizza id (e.g. "DNbUoMtG3H") or model page URL.
        dest_dir: absolute directory to save into (created if needed).
        name: file stem / asset role, e.g. "sedan". Defaults to the model title.

    Returns JSON with the path, licence + attribution, and geometry facts (size in file units in
    Z-up, triangle count, textures, animations). Also updates dest_dir/polypizza_assets.json and
    dest_dir/CREDITS.md so licences are tracked automatically.
    """
    try:
        info = client.lookup(model_id)
        data = client.download_glb(info)
    except PolyPizzaError as e:
        return json.dumps({"ok": False, "error": str(e),
                           "advice": "Download failed: generate this object procedurally instead."})
    dest = Path(dest_dir).expanduser()
    if not dest.is_absolute():
        return json.dumps({"ok": False, "error": "dest_dir must be an absolute path"})
    dest.mkdir(parents=True, exist_ok=True)
    stem = _NAME_RE.sub("_", name or info.title).strip("_").lower() or info.id
    path = dest / f"{stem}.glb"
    path.write_bytes(data)
    try:
        from .glb import inspect_glb

        geo = inspect_glb(data)
    except Exception as e:  # noqa: BLE001
        geo = {"ok": False, "error": f"inspection failed: {e}"}
    mpath = dest / "polypizza_assets.json"
    try:
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    except ValueError:
        manifest = {}
    manifest[stem] = {**info.to_dict(), "file": path.name, "downloaded": time.strftime("%Y-%m-%d"),
                      "size_zup_file_units": geo.get("size_zup_xyz"), "triangles": geo.get("triangles")}
    mpath.write_text(json.dumps(manifest, indent=1))
    credits = _write_credits(dest, manifest)
    return json.dumps({"ok": True, "path": str(path), "bytes": len(data), "model": info.to_dict(),
                       "geometry": geo, "manifest": str(mpath), "credits": str(credits)}, indent=1)


@mcp.tool(structured_output=False)
def server_status() -> str:
    """Access mode (official API or keyless website), rate limit and cache location."""
    cache = client.CACHE_DIR
    counts = {k: len(list((cache / k).glob("*"))) if (cache / k).exists() else 0
              for k in ("search", "model", "glb", "preview")}
    return json.dumps({
        "mode": client.mode(),
        "site_delay_s": client.SITE_DELAY_S,
        "cache_dir": str(cache),
        "cached": counts,
        "api_key_hint": "Set POLY_PIZZA_API_KEY (free at https://poly.pizza/settings/api) for faster official API access.",
    }, indent=1)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
