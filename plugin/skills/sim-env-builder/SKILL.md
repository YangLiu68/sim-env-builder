---
name: sim-env-builder
description: Build a 3D simulated environment (robotics / RL / autonomy scenes - self-driving streets, robot arms, drones, legged robots, warehouses, homes, farms, marine, space, sports, multi-agent) in MuJoCo or NVIDIA Isaac Sim from a description. Chooses the backend, confirms the spec with the user, sources 3D models from Poly Pizza (polypizza MCP) with procedural fallbacks, builds a tested scene + Gym-style env, and smoke-tests rendering, visuals and physics. Use whenever the user asks to create, generate, build or set up a simulation environment, sim scene, virtual world, RL/robot environment or digital twin.
---

# Sim Env Builder

Goal: from one description, deliver a working, physically sound, good-looking simulated
environment in as few user turns as possible (ideally: request -> one confirmation -> done).
Quality bar: every smoke test passes, you have LOOKED at the renders and they match the request.

`SKILL_DIR` below = the directory of this file. Python helpers live in `SKILL_DIR/scripts/simkit`.

## Phase 0 - Setup (1 minute)

```bash
PY=$(bash SKILL_DIR/scripts/setup_env.sh)       # python with mujoco, trimesh, usd-core, gymnasium
$PY SKILL_DIR/scripts/check_platform.py           # OS, GPU, Isaac availability
```
Pick the environment folder: `./<short_snake_name>/` in the current directory unless the user named one.

## Phase 1 - Choose the backend (read reference/backend_choice.md)

Isaac Sim needs an NVIDIA RTX GPU on Linux/Windows. MuJoCo runs anywhere.
- MuJoCo: default. Contact-rich manipulation and locomotion, fast iteration, CPU/laptop/macOS,
  small-to-medium scenes, model-based control, reproducible research, MJX/Warp for GPU RL.
- Isaac Sim: photoreal RTX rendering or synthetic data (Replicator), RTX lidar/camera sensor
  realism, thousands of parallel envs in Isaac Lab, large USD/Omniverse scenes, user already on Isaac.
- If Isaac is the better fit but `isaac_runnable_here` is false, say so in the confirmation and
  offer: build + fully test in MuJoCo here AND export a USD stage + Isaac smoke script for the
  GPU machine (reference/isaac.md). Never claim Isaac results you did not run.

## Phase 2 - Confirm with the user BEFORE building (reference/confirmation.md)

1. First do a quick asset scan so the brief is grounded: one `find_assets` call (polypizza MCP)
   listing every object role, 1-3 queries each.
2. Send ONE short brief (template in reference/confirmation.md): backend + why, what will be in the
   scene, the robot/agent and its control interface, which assets come from Poly Pizza vs
   procedural, key physics (gravity, friction, timestep), deliverables, and 2-4 questions that
   each carry your recommended default.
3. Ask with AskUserQuestion if that tool exists. Otherwise end your turn with the brief as your
   message and wait. Do not build before the user answers.
   Exception: the user said not to ask / "just build it" -> write assumptions into SPEC.md, proceed.
4. Record the confirmed decisions in `<env>/SPEC.md` (template in reference/confirmation.md).

## Phase 3 - Assets: Poly Pizza first, procedural fallback (reference/assets.md)

For every object role:
1. `find_assets` / `search_models` -> read the verdict (found / weak / not_found).
2. found or weak -> `preview_models([...ids])` and LOOK: right object? style consistent? reject
   toys, cartoons in realistic scenes, multi-object bundles, wrong scale class.
3. Accept -> `download_model(id, "<abs env path>/assets/polypizza", "<role>")` (credits are
   recorded automatically in `assets/polypizza/CREDITS.md`).
4. not_found, rejected, or download error -> procedural: `simkit.procedural` recipes (trees,
   buildings, furniture, street props, gates, tableware, terrain...) or compose your own from
   primitives / `lathe` / `extrude` / `rock`. Never leave a requested object out.
5. Write `<env>/assets/ASSETS.md`: role | source (Poly Pizza id + title + licence, or procedural
   recipe) | real size | why. If a model's title differs from the role ("Park Bench" used as a
   picnic table), say so and why it still fits - or search again / go procedural.
   Also list REJECTED candidates (id + reason: "cartoon style", "bundle of 12 objects") so the
   search is traceable even when everything ends up procedural.
Poly Pizza models are static meshes: articulated robots and vehicles come from `simkit.robots`
(you may skin them with a Poly Pizza mesh as visual geoms).

## Phase 4 - Build (reference/robots.md, templates/)

```bash
mkdir -p <env> && cp -r SKILL_DIR/scripts/simkit <env>/simkit       # vendored: env is self-contained
cp SKILL_DIR/templates/build_env.py SKILL_DIR/templates/env.py SKILL_DIR/templates/view.py <env>/   # edit the first two
```
- `build_env.py` regenerates everything deterministically (seeded) -> `scene.xml`.
  Use `simkit.scene.Scene` (sky, ground, lights, look-at cameras, sections for raw MJCF),
  `simkit.assets.glb_to_mujoco(glb, env, name, target={"height": m}, yaw_deg=..)` for Poly Pizza
  models (real-world size, Z-up, per-colour parts, box/hull/parts collision), `simkit.procedural`
  for fallbacks and terrain, `simkit.robots` for tested robots (quadrotor, car, diff_drive, arm
  + gripper, quadruped) and their controllers.
- Real-world scale is mandatory (reference/assets.md size table). Things rest on their support.
- Every object class the user named must be clearly visible at a readable size in the overview
  (a "rocky trail" needs rocks you can see, not 7 cm specks; "rough terrain" needs bumps visible
  at robot scale). The smoke test lists specks under `hints`.
- At least 2 named cameras: an overview and one framed on the subject (the robot/task object fills
  roughly 30-50% of the frame; robot-mounted if any). Fixed
  sensor cameras must not be blocked by the robot's home pose (smoke flags > 15% coverage).
- `world.py` (when the scene has its own dynamics): `make_world(model)` -> `world(model, data)`,
  called before EVERY physics step by smoke, examiner, showcase, env.py, view.py and the dashboard.
  Buoyancy/drag, wind, conveyors, traffic-light cycles, scripted pedestrians and NPC cars live here,
  never inside controller.py (a passive test or an RL policy would otherwise run a world where
  boats sink and lights never change). Task checks that step the sim must call it too.
- `controller.py`: a scripted demo policy (`make_controller(model)` -> `ctrl(model, data)`) that
  shows the environment doing its job (drone flies the gates, car drives the route, arm reaches the object...).
  Set `DEMO_SECONDS = <length of the full demo>` at module level: the smoke test then films and
  checks the WHOLE demo (pick, lift, place / every gate), not just the first 4 seconds.
- `task_checks.py`: functions `check_<name>(model, data) -> {"status", "value", "expect", "details"}`
  for task-level physics. One check per success criterion the user stated or confirmed (name it
  after their words: "feet in contact" -> `check_feet_in_contact`), plus spawn sanity (robot starts
  on its support). At least 2. Write the thresholds AND each metric's exact definition (reference
  point, units: "rover base centre to lander centre, metres") into SPEC.md BEFORE the first run and keep
  them: a narrow miss means fix the controller/scene, never loosen the check (any threshold
  change must appear under "Changes from the confirmed plan"). A check passing within 15% of its limit
  is reported with its margin ("tilt 25.8 deg vs 30 deg limit"). Any behaviour you claim
  (obstacle slow-down, traffic-light cycle, buoyancy, gripper hold) needs a check that exercises
  it, e.g. put an obstacle in the path and assert the robot stops. Every key physical property
  of the scene gets a measured check with a physical target: a ball's rolling deceleration, a
  boat's waterline, a rover's slope climb, low gravity's fall time, gripper holding force.
  A required sensor gets a check too: render the robot camera during the demo and assert a
  non-blank image that shows the task (e.g. tree rows ahead); for lidar, assert sensible ranges.
  Run each task check on the scripted rollout AND 2-3 perturbed ones (other seed, start pose,
  initial speed, light phase) - a check that only holds for one exact rollout proves little.
  Grasp checks require contact between BOTH finger pads and the object throughout lift and carry
  (read `data.contact`), not just "the object went up".
- Colliders: simple and honest. Trees: trunk collider, canopy visual-only unless the user wants
  canopy hits. Multirotors: arm capsules or a hull, not one box that swallows the props.
  Non-convex props (microscope, chair, arch, shelf) get `collision="parts"` or a few primitives;
  one convex hull around them blocks space that is visibly empty (check collision_sheet.png).
- `env.py`: Gymnasium-style `reset/step/render` with task observation, reward stub, termination.
  Its `__main__` self-test drives the env THROUGH the env API with the demo controller (or a
  reference policy) and prints whether the task goal is reached - proof the goal is reachable.
- Debug/camera-rig bodies and markers: geom group 4-5 (never rendered), not in data cameras.
- Suspended things (ceiling lights, hanging signs, wall shelves): name them `mounted_*` or
  `hanging_*` so the floating-object check skips them.
- Lighting: neutral white (no tinted lights); at most 2-3 extra lights; check a white wall stays
  white-grey in the renders.
- No unrequested dressing beyond what makes the scene read correctly; list optional extras in the
  brief so the user can opt in.
- Time box: note the time when you start building (`date`). Bonus/stretch items ("folding is a
  bonus") come only after every must-have passes, and get dropped first. About 60 minutes into
  the build, stop tuning whatever is left: make REPORT.md and README results final and report
  (dropped items listed). A build that never reports is worse than one with a known gap.
- Budget: once every must-have exists and all checks pass, stop adding extras - finish the visual
  check, README and report. Extras go in the README as ideas.
- Run `$PY build_env.py` after every change; compile errors carry line numbers.

## Phase 5 - Smoke tests (reference/smoke_tests.md). All three are required.

```bash
cd <env> && $PY -m simkit.smoke scene.xml --out smoke --controller controller.py --task-checks task_checks.py
```
1. Render test - every named camera + auto views render non-blank, lit, detailed frames;
   segmentation shows which objects are visible (unseen objects are flagged).
2. Physics examiner - solver/contacts/friction/gravity probes with the scene's own settings,
   rollout stability, penetration, fall-through, settling, actuator step responses, determinism,
   timestep sensitivity, real-time factor, layout (floating / sunken / interpenetrating props),
   plus your task checks.
3. Visual check - YOU must open and look at `smoke/contact_sheet.png`, `smoke/filmstrip.png`,
   `smoke/collision_sheet.png` and the closeups in `smoke/renders/` (Read tool), then fill the
   checklist into `smoke/visual_check.md` with concrete observations.
Fix every FAIL. Investigate every WARN (fix it, or explain in README why it is expected).
Re-build and re-run until the verdict is PASS / PASS_WITH_WARNINGS-explained AND the visual
check is honestly all "yes". Expect 2-3 iterations; the first render is rarely right.
Also run `$PY env.py` (random actions, finite observations).

## Phase 6 - Isaac Sim deliverable (only if Isaac was chosen; reference/isaac.md)

`$PY -m simkit.usd_export scene.xml isaac/scene.usda` (export + local validation), copy
`SKILL_DIR/templates/isaac_smoke.py` to `isaac/`, document how to run it on the GPU machine.
State clearly what was and was not executed.

## Phase 7 - Deliver

`<env>/README.md`: what is in the scene, how to run (build, smoke, env, viewer:
`$PY view.py`, which runs the demo controller live; plain `mujoco.viewer` shows passive physics only), cameras, actuators, controller, task checks, asset
credits, known limitations. Final message to the user: verdict line, what was built, which assets
came from Poly Pizza vs procedural, smoke results, paths to `smoke/showcase.mp4` (the video people
watch), contact sheet and filmstrip, anything not done or not verified, and a "Changes from the
confirmed plan" list (swapped assets or licences, changed sizes/counts, dropped extras) - empty
if none, but always present. Copy every number (checks passed, scores, errors) from
`smoke/smoke_report.json` and `physics_report.json`; paste its `summary_line` verbatim
("Smoke PASS: examiner 19 pass / 0 warn / 0 fail; task checks 9/9 pass; render ok") and never write
counts from memory.
Start `<env>/REPORT.md` as soon as the first smoke run is done and update it after each fix, so an
interrupted session still leaves a report; the final message is its last version.

## Deliverable layout (keep exactly; tools and dashboards depend on it)

```
<env>/
  SPEC.md  README.md  REPORT.md  build_env.py  scene.xml  env.py  view.py  controller.py  task_checks.py
  world.py                      (only if the scene has its own dynamics)
  simkit/                       vendored helpers
  assets/polypizza/*.glb  assets/polypizza/CREDITS.md  assets/polypizza/polypizza_assets.json
  assets/meshes/  assets/procedural/  assets/ASSETS.md
  smoke/  (showcase.mp4 contact_sheet.png filmstrip.png collision_sheet.png renders/ rollout.mp4
           physics_report.json render_report.json smoke_report.json env_manifest.json
           SMOKE.md visual_check.md)
  isaac/  (only for Isaac: scene.usda, isaac_smoke.py, README section)
```

## Hard-won rules (each one has broken a real build)

- Stay inside the confirmed scope. The demo shows what the user agreed to (e.g. "standing is
  enough" means a stable stand, not a new locomotion controller). Spend the time on the scene,
  the assets, the checks and the visual review instead; list stretch ideas in the README.
- `<visual>` settings: `sc.set_visual("map", znear=.., zfar=..)` etc. Each element may appear once.

- Never hand-compute camera `xyaxes`: use `Scene.camera(name, pos, target)` / `simkit.scene.lookat`.
- Never hand-write keyframe qpos lists: set `scene.home[joint] = value` (robots do it for you);
  `Scene.write` builds the `home` keyframe by joint name and sets position-actuator ctrl to match.
- Flat/thin meshes need `inertia="shell"` (glb_to_mujoco does it). Visual geoms: `contype=0
  conaffinity=0 group=2 mass=0`; collision proxies in group 3.
- A robot welded to the world collides with its own base: add `<exclude>` for base/first link.
- Position-actuator stiffness: keep omega*dt < 0.6 (omega = sqrt(kp / joint inertia)); add
  armature (0.01-0.5) and damping; use `integrator="implicitfast"` (Scene default).
- Comparing MuJoCo enums with numpy ints: `int(x) == int(mujoco.mjtX.mjY)`; `x in (enum, ...)` silently fails.
- Free objects must start resting on (not inside, not above) their support, else the settling check fails.
- Poly Pizza facing is arbitrary: after conversion render a closeup; fix with `yaw_deg`.
- Big scenes: keep ngeom < ~5000 and meshes < ~50k triangles total for real-time.
- MuJoCo 3.x/4.x: `mujoco.mj_fullM(m, d, M)`; `MjData.qM` was renamed `M`.
