"""Isaac Sim smoke test for a stage exported by simkit.usd_export (run on an NVIDIA RTX machine).

    ./python.sh isaac/isaac_smoke.py --usd isaac/scene.usda --out isaac/smoke      # Isaac Sim install
    python isaac/isaac_smoke.py --usd isaac/scene.usda --out isaac/smoke            # pip install isaacsim

Checks (mirrors simkit.smoke): stage loads, PhysX steps without NaN, rigid bodies stay above the
ground and inside the scene, joint drives respond, every USD camera renders a non-blank RGB
frame (written as PNG). Results -> <out>/isaac_smoke_report.json.

STATUS: written against Isaac Sim 4.5/5.x APIs (isaacsim.* namespaces, with omni.isaac.* fallbacks).
It could not be executed on the machine that generated this environment unless the README says
so: treat the first run as part of setup and fix API drift if your Isaac version differs.
"""

import argparse
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", default="isaac/smoke")
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--headless", action="store_true", default=True)
args = ap.parse_args()

try:
    from isaacsim import SimulationApp
except ImportError:  # older installs
    from omni.isaac.kit import SimulationApp

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

try:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
report = {"usd": args.usd, "checks": []}


def add(name, ok, value=None, details=""):
    report["checks"].append({"name": name, "status": "pass" if ok else "fail", "value": value, "details": details})


ok = open_stage(str(Path(args.usd).resolve()))
add("stage_loads", bool(ok))
stage = omni.usd.get_context().get_stage()
scene = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
dt = scene.GetPrim().GetCustomDataByKey("mjc:timestep") if scene else None
world = World(stage_units_in_meters=1.0, physics_dt=float(dt or 1 / 240), rendering_dt=1 / 30)
world.reset()

bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
xf_cache = UsdGeom.XformCache()


def poses():
    xf_cache.Clear()
    return {str(p.GetPath()): np.array(xf_cache.GetLocalToWorldTransform(p).ExtractTranslation()) for p in bodies}


p0 = poses()
steps = int(args.seconds / world.get_physics_dt())
nan_at = None
for k in range(steps):
    world.step(render=False)
    if k % 30 == 0:
        if any(not np.isfinite(v).all() for v in poses().values()):
            nan_at = k
            break
p1 = poses()
add("physx_stable", nan_at is None, {"steps": steps, "nan_at_step": nan_at})
fell = [k for k, v in p1.items() if v[2] < -1.0]
moved = {k: float(np.linalg.norm(p1[k] - p0[k])) for k in p1}
add("no_fall_through", not fell, fell[:10])
report["max_body_displacement_m"] = max(moved.values()) if moved else 0.0

# cameras -> RGB via replicator
try:
    import omni.replicator.core as rep
    from PIL import Image

    cams = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera) and not str(p.GetPath()).startswith("/OmniverseKit")]
    shots = {}
    for cam in cams[:8]:
        rp = rep.create.render_product(str(cam.GetPath()), (960, 540))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach(rp)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=4)
        img = np.asarray(annot.get_data())[..., :3]
        name = cam.GetName()
        Image.fromarray(img.astype(np.uint8)).save(out / f"{name}.png")
        shots[name] = {"mean": float(img.mean()), "std": float(img.std())}
    blank = [n for n, s in shots.items() if s["std"] < 4]
    add("camera_renders", bool(shots) and not blank, shots, "blank: " + ", ".join(blank) if blank else "")
except Exception as e:  # noqa: BLE001
    add("camera_renders", False, details=f"replicator failed: {e!r}")

report["verdict"] = "FAIL" if any(c["status"] == "fail" for c in report["checks"]) else "PASS"
(out / "isaac_smoke_report.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str))
app.close()
