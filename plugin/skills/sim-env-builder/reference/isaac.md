# Isaac Sim path

## When the GPU machine is NOT this machine (most common)

1. Build and smoke-test the scene in MuJoCo exactly as in the main workflow (this is the tested source of truth).
2. Export: `$PY -m simkit.usd_export scene.xml isaac/scene.usda`
   - bodies -> rigid bodies (mass, COM, inertia), geoms -> USD prims with displayColor, colliders,
     joints (revolute/prismatic/spherical) with limits, position actuators -> joint drives (stiffness/damping),
     fixed-base robots -> fixed joint to world, articulation roots, cameras, sun + dome light, gravity.
   - The command also validates the stage locally (usd-core): masses, nested rigid bodies, joint targets, bounds.
   - Site/thrust actuators (drones) are exported as metadata (`/World` customData `mjc:site_actuators`):
     apply forces in an Isaac controller script.
3. `cp SKILL_DIR/templates/isaac_smoke.py isaac/` and document in README:
   `./python.sh isaac/isaac_smoke.py --usd isaac/scene.usda --out isaac/smoke` (or `python` with pip isaacsim).
4. Tell the user plainly: "USD exported and validated structurally; PhysX/RTX not executed here".

## When Isaac Sim runs here (Linux/Windows + RTX + isaacsim installed)

- Same steps 1-2, then run `isaac_smoke.py` and include its renders in the visual check.
- For large photoreal scenes, author directly in USD after export: replace displayColor with
  MDL/UsdPreviewSurface materials, reference GLB->USD assets (`simkit.assets.glb_to_usd`), add
  Replicator randomisation. Re-run isaac_smoke.py after every change.
- Isaac Lab (parallel RL): wrap the stage as an `InteractiveScene` asset config; keep the MuJoCo
  env.py as the reference implementation of observation/action semantics.

## Alternatives

- Isaac Sim's own MJCF importer (`isaacsim.asset.importer.mjcf`) imports robots from MJCF directly;
  good for articulated robots, weaker for whole scenes with meshes/hfields.
- GLB/OBJ props can be converted inside Isaac with `omni.kit.asset_converter`.
