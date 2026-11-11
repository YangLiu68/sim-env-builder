"""Build the environment from scratch (template). Deterministic: same SEED -> same scene.

    python build_env.py            # writes scene.xml (+ meshes under assets/)

Structure: 1) assets (Poly Pizza GLBs already downloaded into assets/polypizza by the MCP,
converted here; procedural fallbacks for the rest), 2) layout, 3) robots/agents, 4) cameras,
5) write + compile check. Keep every number that matters (sizes, positions, gains) named.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # vendored simkit/

from simkit import procedural as P  # noqa: E402
from simkit import robots as R  # noqa: E402
from simkit.assets import glb_to_mujoco  # noqa: E402
from simkit.scene import Scene  # noqa: E402

SEED = 0
rng = np.random.default_rng(SEED)
PP = HERE / "assets/polypizza"


def pp_asset(name, **kw):
    """Poly Pizza model if its GLB was downloaded, else None (caller uses a procedural fallback)."""
    glb = PP / f"{name}.glb"
    return glb_to_mujoco(glb, HERE, name, **kw) if glb.exists() else None


sc = Scene("my_env", timestep=0.002)
sc.sky("day")
sc.ground(size=30, style="grid")

# ---- assets: Poly Pizza first, procedural fallback ----------------------------------------
tree = pp_asset("tree", target={"height": 5.0}, collision="hull")
for i in range(6):
    x, y = rng.uniform(-12, 12, 2)
    if tree is not None:
        sc.add_asset(tree)
        sc.body(tree.body_xml(f"tree{i}", pos=(x, y, 0), euler=(0, 0, rng.uniform(0, 360))))
    else:
        sc.body(P.tree(5.0, "round", seed=i).body_xml(f"tree{i}", pos=(x, y, 0)))

# ---- robot -----------------------------------------------------------------------------
robot = R.quadrotor("drone", pos=(0, 0, 0.12))
robot.add_to(sc)

# ---- cameras ---------------------------------------------------------------------------
sc.camera("overview", pos=(10, -10, 7), target=(0, 0, 1))

sc.write(HERE / "scene.xml")
print("wrote", HERE / "scene.xml")
