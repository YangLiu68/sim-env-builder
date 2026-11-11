"""Screenshot dashboard pages with the installed Chrome (headless). Usage: ui_shots.py OUTDIR URL#route ..."""
import sys, time
from playwright.sync_api import sync_playwright
out = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1500, "height": 920}, device_scale_factor=1)
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    for i, url in enumerate(sys.argv[2:]):
        wait = 6
        if "@" in url:
            url, wait = url.split("@"); wait = float(wait)
        pg.goto(url); time.sleep(wait)
        path = f"{out}/shot{i}.png"; pg.screenshot(path=path); print("saved", path)
    print("\n".join(logs[:20]) or "no console errors")
    b.close()
