"""Talk to the Poly Pizza MCP over stdio exactly like Claude Code does."""
import asyncio, json, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]

async def main(dest):
    params = StdioServerParameters(command=str(ROOT / "plugin/bin/polypizza-mcp"), args=[])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print("server:", init.server_info.name, "| instructions:", (init.instructions or "")[:80])
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            res = await s.call_tool("find_assets", {"needs": [
                {"role": "drone", "queries": ["drone", "quadcopter"]},
                {"role": "racing_gate", "queries": ["racing gate", "drone gate"]},
            ]})
            out = json.loads(res.content[0].text)
            print(json.dumps(out["summary"]), {k: (v["verdict"], [c["title"] for c in v["candidates"]]) for k, v in out["roles"].items()})
            best = out["roles"]["drone"]["candidates"][0]["id"]
            res = await s.call_tool("preview_models", {"model_ids": [c["id"] for c in out["roles"]["drone"]["candidates"]]})
            print("preview content types:", [c.type for c in res.content])
            img = [c for c in res.content if c.type == "image"][0]
            import base64; Path(dest, "grid.png").write_bytes(base64.b64decode(img.data))
            res = await s.call_tool("download_model", {"model_id": best, "dest_dir": dest, "name": "drone"})
            d = json.loads(res.content[0].text)
            print("download:", d["ok"], d.get("path"), d.get("geometry", {}).get("size_zup_xyz"), d.get("geometry", {}).get("triangles"), d["model"]["attribution"])

asyncio.run(main(sys.argv[1]))
