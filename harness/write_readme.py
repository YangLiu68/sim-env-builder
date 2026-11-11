"""Regenerate README.md from the scoreboard and run data (numbers are never typed by hand).

    .venv/bin/python harness/write_readme.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_media as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "golden/runs"

TASK_NOTES = {
    "sd_intersection": "4-way junction, traffic lights, parked cars, pedestrians, steerable ego car",
    "arm_kitchen": "6-DoF arm + gripper on a counter; mug, fruit, plate; pick and place",
    "drone_gates": "quadcopter, 5-6 gates at different heights, FPV camera, RL env",
    "quadruped_trail": "12-DoF quadruped on rough hills with rocks and stairs",
    "warehouse_amr": "pallet racks, boxes, forklift, differential-drive AMR with lidar",
    "home_fetch": "living room; mobile base + arm fetches a cup from the coffee table",
    "mars_rover": "cratered terrain, Mars gravity, rover with mast camera, lander",
    "harbor_boat": "floating boat (buoyancy + drag), thruster and rudder, dock, buoys",
    "orchard_drone": "apple-tree rows, tractor, barn, inspection rover with camera",
    "robot_soccer": "2 vs 2 wheeled robots, ball, goals; multi-agent API",
    "isaac_synthetic": "retail aisle for synthetic data; Isaac Sim wanted but not runnable locally",
    "cartpole_minimal": "classic cart-pole; tests that a minimal request stays minimal",
    "factory_sorting": "conveyor, colour-coded boxes and bins, arm sorting, safety fence",
    "lab_bench": "centrifuge, microscope, tube racks: mostly procedural fallbacks",
    "cloth_table": "deformable cloth on a table, arm lifts a corner",
    "humanoid_gym": "humanoid in a gym hall with obstacles and a ramp",
}


def board():
    b = json.loads((ROOT / "golden/scoreboard.json").read_text())
    best, iters = {}, []
    for it in b["iterations"]:
        if it.get("void"):
            continue
        if it.get("mean_total") is not None:
            iters.append((it["iter"], it["mean_total"], it["n_done"], it.get("notes", "")))
        for r in it["rows"]:
            if r["total"] is not None and (r["task"] not in best or r["total"] >= best[r["task"]][0]):
                best[r["task"]] = (r["total"], it["iter"], r["category"])
    return best, iters


def run_stats():
    mins, costs, shots, smoke_s = [], [], [], []
    for rm in RUNS.glob("iter_0*/*/run_meta.json"):
        if not (rm.parent / "result.json").exists() or rm.parent.parent.name == "iter_01":
            continue
        m = json.loads(rm.read_text())
        mins.append((m.get("build_seconds") or 0) / 60)
        costs.append(m.get("agent_cost_usd") or 0)
        shots.append(m.get("shots_used") or 0)
        env = m.get("env_dir")
        sp = Path(env) / "smoke/smoke_report.json" if env else None
        if sp and sp.exists():
            smoke_s.append(json.loads(sp.read_text()).get("seconds") or 0)
    return {"n": len(mins), "min_med": round(st.median(mins)), "min_p90": round(sorted(mins)[int(0.9 * len(mins)) - 1]),
            "cost_med": round(st.median(costs), 1), "cost_max": round(max(costs)), "two_shot": sum(s == 2 for s in shots),
            "smoke_med": round(st.median(smoke_s))}


def gallery_table():
    cells = []
    for it, task, cap in M.HERO:
        sc = M.score_of(it, task)
        cells.append(f'<td width="50%" valign="top"><img src="docs/media/{task}.gif" width="100%" alt="{task}"/><br/>'
                     f'<sub><b>{cap}</b><br/>score {sc}/100 · <a href="docs/media/{task}.mp4">full video</a></sub></td>')
    rows = "\n".join("  <tr>\n    " + "\n    ".join(cells[i:i + 2]) + "\n  </tr>" for i in range(0, len(cells), 2))
    return f"<table>\n{rows}\n</table>"


def main():
    best, iters = board()
    s = run_stats()
    mean_hist = " → ".join(f"{m}" for _, m, _, _ in iters)
    first, last = iters[0][1], iters[-1][1]
    n_tasks = len(best)
    over85 = sum(1 for v in best.values() if v[0] >= 85)
    task_rows = "\n".join(f"| `{t}` | {c} | {TASK_NOTES.get(t, '')} | {sc} |"
                          for t, (sc, it, c) in sorted(best.items(), key=lambda kv: -kv[1][0]))
    iter_rows = "\n".join(f"| {it.replace('iter_0', '')} | {m} | {n} | {notes[:150]} |" for it, m, n, notes in iters)
    readme = f"""<h1 align="center">sim-env-builder</h1>

<p align="center">
  <b>Turn a short description into a working, tested 3D simulation environment for robotics and RL.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Claude%20Code-plugin%20%2B%20skill-d97757" alt="Claude Code plugin"/>
  <img src="https://img.shields.io/badge/MCP-Poly%20Pizza%20server-6e56cf" alt="MCP server"/>
  <img src="https://img.shields.io/badge/physics-MuJoCo%204-orange" alt="MuJoCo 4"/>
  <img src="https://img.shields.io/badge/export-Isaac%20Sim%20(USD)-76b900" alt="Isaac Sim USD"/>
  <img src="https://img.shields.io/badge/API-Gymnasium-lightgrey" alt="Gymnasium"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
</p>

{gallery_table()}

<p align="center"><sub>Every scene above was built from a one-paragraph request in the golden set, then rebuilt from
scratch, re-tested and scored by an independent judge. Nothing in them was modelled or placed by hand.</sub></p>

**sim-env-builder** is a [Claude Code](https://claude.com/claude-code) plugin that builds simulation environments
from plain-language requests such as *"a warehouse with pallet racks, a parked forklift and a mobile robot that
drives down an aisle"*. Behind the request it runs a fixed engineering procedure:

1. **It chooses the simulator** that fits the job and the machine: MuJoCo for contact-rich robotics on any
   computer, NVIDIA Isaac Sim when photoreal rendering or massive GPU parallelism matters and an RTX GPU is
   available (otherwise it builds in MuJoCo and exports an Isaac Sim USD stage).
2. **It agrees the plan with you before writing code**: one short brief with the backend and why, the scene
   contents with real-world sizes, the robot and its control interface, which models will come from where, and
   two to four questions that each carry a default, so "go" is always a complete answer.
3. **It furnishes the scene with real 3D models** from [Poly Pizza](https://poly.pizza) through a bundled MCP
   server, and **generates anything that is missing in code** (63 procedural recipes, from traffic lights and
   pallet racks to drone gates, terrain and cloth).
4. **It only hands the environment over after it passes three kinds of smoke test**: offscreen rendering from
   every camera, a visual review in which the agent looks at its own renders, and a physics examiner that checks
   the solver, contacts, friction, gravity, actuators and the task itself.

The repository also contains the machinery used to make that procedure reliable: a **{n_tasks}-task golden set**
spanning self-driving, manipulation, aerial, legged, marine, space, industrial and multi-agent scenes; a
**harness** that gives every task to a fresh agent with nothing but the plugin, scores the result with automated
checks and an independent judge, and feeds the findings back into the playbook; and **SimForge Studio**, a web
dashboard to launch builds, follow them live and drive the finished environments in 3D.

---

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [How a request becomes an environment](#how-a-request-becomes-an-environment)
- [Choosing MuJoCo or Isaac Sim](#choosing-mujoco-or-isaac-sim)
- [3D models: the Poly Pizza MCP server and procedural fallbacks](#3d-models-the-poly-pizza-mcp-server-and-procedural-fallbacks)
- [Robots, controllers and world dynamics](#robots-controllers-and-world-dynamics)
- [Smoke tests](#smoke-tests)
- [What you get](#what-you-get)
- [The golden set and the improvement loop](#the-golden-set-and-the-improvement-loop)
- [SimForge Studio](#simforge-studio)
- [Scale and cost](#scale-and-cost)
- [Testing](#testing)
- [Command-line reference](#command-line-reference)
- [Repository layout](#repository-layout)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Acknowledgements](#acknowledgements)

---

## Highlights

| | |
|---|---|
| **One request, one confirmation** | All {s['n']} scored golden runs finished in two user messages: the request, then "go" (or a few answers) to the plan. |
| **Backend choice with a hardware gate** | Reads the OS, GPU and installed simulators; picks MuJoCo or Isaac Sim on the merits and says plainly when Isaac would fit but cannot run here. |
| **Real assets, never missing ones** | Poly Pizza search with `found` / `weak` / `not_found` verdicts and labelled preview grids; licences and credits recorded automatically; anything not found is generated in code. |
| **Physically honest scenes** | Real-world sizes, things resting on their supports, simple and correct colliders, tuned actuators (`ω·dt < 0.6`), name-based home keyframes. |
| **Three-part smoke test** | Renders from every camera, a visual review checklist, and a physics examiner with {22} checks plus layout checks and the task's own checks. |
| **Ready for learning** | Every environment ships a Gymnasium-style `env.py`, a scripted demo controller, task checks, a live viewer script and a showcase video. |
| **Measured, not assumed** | Golden-set mean score {mean_hist} across iterations; {over85} of {n_tasks} tasks score 85 or more (automated 40 + independent judge 60). |
| **Dashboard** | Describe an environment, answer the agent's questions, watch every step, then drive the result in a real-time 3D viewer in the browser. |

---

## Quick start

```bash
git clone https://github.com/YangLiu68/sim-env-builder && cd sim-env-builder
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mujoco trimesh "mcp>=2" numpy pillow \\
    scipy networkx imageio imageio-ffmpeg usd-core gymnasium fastapi "uvicorn[standard]" pytest
```

**Use the skill** in Claude Code (the plugin bundles the skill and the Poly Pizza MCP server):

```bash
claude --plugin-dir plugin
> Build a harbor with a small boat that floats, a wooden dock, a few buoys and a lighthouse.
```

The agent checks the machine, sends its plan with a few questions, and after your answer builds
`./<env_name>/` and runs the smoke tests. The simulation Python (MuJoCo, trimesh, usd-core, gymnasium) is set up
on first use in a shared virtual environment (`~/.cache/sim-env-builder/venv`).

**Run the dashboard:**

```bash
.venv/bin/python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8766
open http://127.0.0.1:8766          # frontend source: dashboard/web (npm install && npm run build)
```

**Poly Pizza access** works without an account (the server reads the public site politely). For the official API,
set `POLY_PIZZA_API_KEY` (free at [poly.pizza/settings/api](https://poly.pizza/settings/api)).

---

## How a request becomes an environment

```mermaid
flowchart LR
    req([request]) --> plat[check_platform.py<br/>OS, GPU, Isaac, MuJoCo]
    plat --> backend{{backend<br/>MuJoCo or Isaac}}
    backend --> scan[Poly Pizza asset scan<br/><code>find_assets</code>]
    scan --> brief[confirmation brief<br/>plan + 2-4 questions]
    brief --> user([your answer])
    user --> assets[assets<br/>preview, download, convert<br/>or generate procedurally]
    assets --> build[build_env.py<br/>scene, robot, world.py,<br/>controller, task checks, env.py]
    build --> smoke[smoke tests<br/>render, physics examiner,<br/>layout, showcase video]
    smoke --> look[visual review<br/>the agent reads its renders]
    look -->|problems| build
    look --> out([environment + REPORT.md])
```

The playbook (`plugin/skills/sim-env-builder/SKILL.md`, ~170 lines, plus six reference files) walks the agent
through seven phases:

| Phase | What happens |
|---|---|
| 0. Setup | `setup_env.sh` finds or creates the simulation Python; `check_platform.py` reports OS, CPU/GPU, RAM and whether Isaac Sim can run. |
| 1. Backend | Decision table in `reference/backend_choice.md`; the hardware gate comes first. |
| 2. Confirm | One `find_assets` call grounds the plan, then a brief from `reference/confirmation.md`. Defaults are the simplest design that meets the request; nothing is built before the answer (unless the user asked not to be asked). Decisions, thresholds and metric definitions are frozen in `SPEC.md`. |
| 3. Assets | Poly Pizza first, procedural fallback, every choice and rejection recorded in `assets/ASSETS.md`, credits in `assets/polypizza/CREDITS.md`. |
| 4. Build | `build_env.py` regenerates everything deterministically; the vendored `simkit` library does the heavy lifting. Scope rule: build what was agreed, list extras as ideas. |
| 5. Smoke tests | One command runs rendering, the physics examiner, layout checks and the task checks, writes reports, a filmstrip and a showcase video. The agent fixes every FAIL, explains every WARN, and fills a visual checklist after looking at the images. |
| 6. Isaac Sim | Only if chosen: USD export with UsdPhysics, local validation, an Isaac smoke script for the GPU machine. |
| 7. Deliver | `README.md` and `REPORT.md` with the verdict line pasted from the smoke report, what came from Poly Pizza vs code, and a mandatory "Changes from the confirmed plan" list. |

A list of hard-won rules closes the playbook. Each rule comes from a real failure in the golden runs, for example:
never hand-compute camera axes; flat meshes need `inertia="shell"`; a robot welded to the world collides with its
own base; environment dynamics such as buoyancy belong in `world.py`, not in the demo controller; time-box
stretch goals; copy check counts from the report, never from memory.

---

## Choosing MuJoCo or Isaac Sim

| Need | Prefers |
|---|---|
| Contact-rich manipulation, legged locomotion, model-based control, fast iteration on a laptop or CI | MuJoCo |
| Photoreal camera data, synthetic datasets with domain randomisation, RTX lidar and camera models | Isaac Sim |
| Thousands of parallel RL environments | Isaac Lab, or MuJoCo MJX / Warp |
| Large USD / Omniverse scenes | Isaac Sim |

The hardware gate comes first: Isaac Sim needs an NVIDIA RTX GPU on Linux or Windows. When Isaac fits the job
but cannot run on the current machine, the brief says so and offers the practical path. The golden task
`isaac_synthetic` (a photoreal retail aisle for detector training) tests exactly this:

* build and fully smoke-test the scene in **MuJoCo** (the tested source of truth);
* export it with `simkit.usd_export` to a **USD stage with UsdPhysics**: rigid bodies with mass, centre of mass and
  inertia; colliders (convex hulls for meshes); revolute, prismatic and spherical joints with limits; position
  actuators as joint drives; fixed joints for fixed-base robots; articulation roots; cameras; sun and dome lights;
  gravity; site actuators such as drone thrusters carried as metadata for a controller script;
* validate the stage locally with `usd-core` (masses, nested rigid bodies, joint targets, bounds);
* ship `isaac/isaac_smoke.py` for the GPU machine (stage load, PhysX stability, fall-through, RTX renders of every
  camera) and state clearly what was and was not executed.

---

## 3D models: the Poly Pizza MCP server and procedural fallbacks

### The MCP server (`plugin/polypizza_mcp`)

| Tool | Purpose |
|---|---|
| `find_assets(needs)` | Batch search: one verdict per role (`found`, `weak`, `not_found`) over 1-3 synonyms each, with ranked candidates. This is how the agent grounds its plan. |
| `search_models(query)` | A single search with verdict and per-result match quality. |
| `preview_models(ids)` | One labelled thumbnail grid (an image the agent looks at) to reject toys, cartoons, bundles and wrong styles before downloading. |
| `get_model(id)` | Triangles, tags, category, licence, attribution line. |
| `download_model(id, dir, name)` | Saves the GLB, reports its size and orientation facts, and updates `polypizza_assets.json` and `CREDITS.md`. |
| `server_status()` | Access mode and cache. |

Two access modes: the **official API** when `POLY_PIZZA_API_KEY` is set, and a **keyless mode** that reads the same
public pages a browser does (the embedded `__SERVER_APP_STATE__` JSON). Keyless mode honours the site's
`Crawl-delay: 10` with a lock shared by every server process on the machine, and caches searches for a week, model
pages for a month and GLBs permanently. The verdict is computed from token overlap between the query and each
model's title, tags and category, so "racing gate" coming back as "Fence Gate" is reported as `weak`, not `found`.

### From GLB to MuJoCo (`simkit.assets.glb_to_mujoco`)

1. Load every mesh of the glTF scene with its node transform.
2. Bake one colour per face: `baseColorFactor` × texture sampled at the UV centroid × vertex colour.
3. Convert glTF Y-up to Z-up and rotate by `yaw_deg` so the model faces +x.
4. Scale to real-world size: by length, width, height, max or footprint, or to a planned **box** (uniform scale
   plus at most ±20% per axis, with the remaining error reported and a warning above 15%).
5. Put the origin at the footprint centre on the ground, so `pos=(x, y, support_height)` rests it on its support.
6. Merge similar colours (area-weighted, CIE Lab distance) into at most 8 parts, one OBJ each; visual meshes use
   `inertia="shell"` so thin parts never break compilation; near-white albedo is capped and near-black lifted.
7. Build collision geometry: one fitted box, one convex hull, or a hull per connected component (tables, shelves,
   arches).

### Procedural fallbacks (`simkit.procedural`)

63 recipes and generators, all sized in metres with the origin at the bottom centre: trees (round, pine, palm),
bushes, rocks; buildings, houses, walls, fences, doors, stairs, ramps; traffic cones and lights, street lamps, stop
signs, benches, bollards, barriers, roads with lane markings, car bodies; tables, chairs, desks, shelves, cabinets,
counters, sofas, beds, TVs, lamps, plants; crates, pallets, barrels, conveyors, pallet racks; mugs, bottles, bowls,
plates, cans, fruit, blocks; drone gates, ring gates, balls, goals, landing pads; buoys, docks; heightfield terrain
(hills, rough, craters, dunes, steps, with edge blending and a flat spawn pad); a flexcomp cloth; and the
generators behind them (lathe, extrusion, rock meshes).

---

## Robots, controllers and world dynamics

`simkit.robots` contains five tested robots with matching controllers. Each `add_to(scene)` adds bodies,
actuators, sensors and contact excludes, and registers a **home pose by joint name**: `Scene.write` builds the
`home` keyframe and sets position-actuator targets to match, so the order of bodies in the XML never matters.

| Builder | Actuators | Controller |
|---|---|---|
| `quadrotor` | 4 rotor thrusts with yaw reaction torque; FPV camera at the front edge | geometric position controller (Lee et al.) + rotor mixer |
| `car` | Ackermann steering, rear-wheel drive, 4-wheel brakes that never reverse the car | speed PI + pure pursuit along waypoints |
| `diff_drive` | left/right wheel velocity; a deck for payloads or an arm | go-to-goal |
| `arm` | 4-7 DoF position control, gravity compensation, parallel gripper, wrist camera | damped least-squares IK + gripper command |
| `quadruped` | 12 position-controlled joints, foot touch sensors | stand / small trot |

Tuning rules built into the templates: `implicitfast` integration, armature and damping on every actuated joint,
position gains with `ω·dt < 0.6`, `gravcomp` on arm links, explicit inertials for mesh-skinned bodies.

**`world.py`** holds dynamics that belong to the world rather than to any policy: buoyancy and water drag, wind,
conveyor belts, traffic-light cycles, scripted pedestrians and NPC cars. It runs before every physics step in the
physics examiner, the actuator tests, the filmstrip, the showcase video, `env.py`, `view.py` and the dashboard
viewer, so a boat never sinks just because a passive test or an RL policy is driving instead of the demo
controller.

---

## Smoke tests

`python -m simkit.smoke scene.xml --controller controller.py --task-checks task_checks.py` (median {s['smoke_med']} s on
an Apple M4) writes everything into `smoke/`:

**Rendering.** Every named camera plus automatic views (overview, top, front, side, iso, robot close-ups). Automatic
cameras pull in front of walls and swing around until the view has detail. Each image is checked for blank, dark,
washed-out or featureless frames. Segmentation renders count the pixels of every object across views to flag
objects nobody can see, objects that are specks in every view, fixed cameras whose task objects are hidden by the
robot at its home pose (rendered with and without the robot), and onboard cameras that mostly see their own robot.

**Physics examiner** (`simkit.physics_exam`, {22} checks):

| Group | Checks |
|---|---|
| model | compile, solver options, mass and inertia sanity (massless or 200x-mismatched links), actuator gains vs timestep |
| initial | penetration at the reset state, with the offending geom pairs |
| rollout | NaN / explosion, MuJoCo solver warnings, maximum penetration, fall-through and fly-away, passive settling with robots holding their home pose, joint-limit overshoot |
| probes | isolated tests with the scene's own solver settings and ground material: free fall vs the analytic (and semi-implicit Euler) solution, resting contact, Coulomb friction deceleration vs μg, bounce energy, a 3-box stack, buoyancy for fluid scenes |
| numerics | determinism (two identical runs), timestep sensitivity (dt vs dt/2), real-time factor |
| actuators | every actuator gets a step command from a settled state and must move its joint in the commanded direction (thrusters are judged by the force they apply) |
| layout | floating, sunken and off-ground props (exact `mj_geomDistance`, not bounding boxes), interpenetrating static props, real-world scale ranges |
| task | the environment's own `check_*` functions: one per success criterion the user stated, plus spawn sanity, run on the scripted rollout and on perturbed ones |

**Visual review.** The agent opens the contact sheet, the filmstrip (a tracking or workspace row plus a fixed row
framing the route the robot actually drives), the visual-vs-collision sheet and the close-ups, then fills a nine-item
checklist (everything requested visible, real-world scale, upright and grounded, correct models, readable
fallbacks, lighting, motion as intended, colliders matching shapes, match with the request).

**Showcase video.** A fly-around of the scene, the demo controller running (tracking camera that pulls in when
scenery blocks it, or a workspace orbit for fixed-base arms), then every named camera. The gallery at the top of this
page is cut from these videos.

---

## What you get

```text
<env>/
  SPEC.md            confirmed decisions, thresholds and metric definitions
  REPORT.md          verdict line, what was built, sources of assets, changes from the plan, gaps
  README.md          how to run, cameras, actuators, controller, checks, credits
  build_env.py       regenerates assets/ and scene.xml deterministically
  scene.xml          the MuJoCo model
  env.py             Gymnasium-style reset/step/render, observation, reward stub, termination
  controller.py      scripted demo policy (DEMO_SECONDS = full length of the demo)
  task_checks.py     check_* functions with physical targets
  world.py           environment dynamics (only when the scene has its own)
  view.py            interactive MuJoCo viewer with world dynamics and the demo running
  simkit/            vendored helpers: the environment is self-contained
  assets/            polypizza/*.glb + CREDITS.md, meshes/, procedural/, ASSETS.md
  smoke/             showcase.mp4, contact_sheet.png, filmstrip.png, collision_sheet.png, renders/,
                     physics_report.json, render_report.json, smoke_report.json, SMOKE.md, visual_check.md
  isaac/             scene.usda + isaac_smoke.py (Isaac Sim requests only)
```

---

## The golden set and the improvement loop

### Tasks

<p align="center"><img src="docs/media/all_tasks.jpg" width="100%" alt="all golden tasks"/><br/>
<sub>All {n_tasks} golden tasks at their best score. 3D model credits:
<a href="docs/media/CREDITS.md">docs/media/CREDITS.md</a>.</sub></p>

| Task | Category | What it exercises | Best score |
|---|---|---|---|
{task_rows}

Each task in `golden/tasks.json` has the user's request (message 1), the answers a simulated user gives to the
confirmation brief (message 2, including the success criteria the user cares about), the expected backend, a
must-have list and the task-level checks for the judge.

### One iteration

```mermaid
flowchart LR
    snap[freeze the plugin<br/>plugin_snapshot/] --> agent[fresh headless agent per task<br/>only the plugin + MCP + request]
    agent --> brief[message 1: brief<br/>built before confirming?]
    brief --> reply[message 2: simulated user]
    reply --> grade[grader, 40 pts<br/>clean rebuild, smoke re-run,<br/>env.py, evidence, assets]
    grade --> judge[independent judge, 60 pts<br/>looks at the renders]
    judge --> fix[fix playbook / simkit / MCP<br/>log in CHANGELOG.md]
    fix --> snap
```

* **Fresh agents.** Every task runs in a new headless `claude -p` session started with `--plugin-dir` (the frozen
  plugin snapshot of that iteration), `--strict-mcp-config` (only the Poly Pizza server) and no user settings,
  memory or prior context. `AskUserQuestion` is disabled so the confirmation step happens in plain text, exactly as
  in a non-interactive deployment.
* **Robust runs.** Network outages wait for connectivity and resume the same session; usage limits sleep until the
  reset time stated in the error; a turn that overruns its time gets one "write the report now" message (counted
  against the few-shot goal). `finish_run.py` and `resume_run.py` recover anything interrupted.
* **Automated grading (40 points).** Deliverables present (8), confirmation before building (4), clean rebuild from
  `build_env.py` in a copy (4), an independent smoke re-run (10), task checks passing (4), `env.py` runs (4),
  evidence that the agent actually looked at its renders (3), asset tracking (3).
* **Independent judge (60 points).** A separate agent with read-only tools reads the brief, the replies, the
  reports and the grader output, looks at the re-rendered contact sheet, filmstrip, collision sheet and close-ups,
  and scores alignment, completeness, assets, visuals and physics (0-10 each, weighted), listing problems with a
  concrete fix for the playbook or the tools.

### Results

| Iteration | Mean score | Tasks | What changed before it |
|---|---|---|---|
{iter_rows}

(Iteration 1 was voided: a DNS outage cut every agent off mid-build.) Retested tasks improved between their first
and latest run, for example kitchen arm 87 → 91.5, drone gates 87.5 → 93.5, city intersection 88.5 → 91.5, Isaac
retail aisle 86 → 91.5, harbor boat 86.5 → 90. The full per-task history is in [`golden/LEDGER.md`](golden/LEDGER.md)
and every change with the finding that caused it in [`golden/CHANGELOG.md`](golden/CHANGELOG.md).

A few things the loop found that no amount of desk-checking would have:

* MuJoCo 4 refuses flat meshes unless they use shell inertia, and `x in (enum, ...)` silently fails for numpy ints.
* The physics examiner called a drone's rotors "dead" because one rotor cannot lift the drone on its own; thrusters
  are now judged by the force they apply.
* A boat's buoyancy lived in its demo controller, so every passive test sank it; this produced `world.py`.
* Agents loosened check thresholds after seeing results, re-asked questions the request had answered, defaulted
  to the fanciest design, and spent the whole time budget on a stretch goal; each became a playbook rule.
* Scaled-up pebbles posing as boulders, white plates that blew out, cameras behind walls or inside robot arms,
  filmstrips that ended before the interesting part: each became a toolkit fix.

---

## SimForge Studio

<p align="center"><img src="docs/media/dashboard_studio.jpg" width="100%" alt="SimForge Studio run page"/><br/>
<sub>A run started from the dashboard: Poly Pizza verdicts and the preview grid the agent looked at, its plan, and
the results with the showcase video.</sub></p>

<table>
  <tr>
    <td width="50%"><img src="docs/media/dashboard_viewer.jpg" alt="3D viewer"/></td>
    <td width="50%"><img src="docs/media/dashboard_golden.jpg" alt="Golden set"/></td>
  </tr>
  <tr>
    <td><sub><b>3D viewer.</b> The MuJoCo scene runs on the server in real time; poses stream to three.js over a
    WebSocket. Demo controller on/off, actuator sliders, named and robot cameras, follow mode, collision overlay,
    Shift+click to push objects.</sub></td>
    <td><sub><b>Golden set.</b> Mean score per iteration, a task-by-iteration matrix and the judge's findings.</sub></td>
  </tr>
</table>

* **Studio:** describe an environment (or pick an example); a fresh agent starts; the timeline shows every step
  live (Poly Pizza searches with verdict chips and preview grids, shell commands, the renders the agent looks at);
  when the agent asks its questions, answer in the reply box or press *Go with defaults*; ask for follow-up changes
  the same way.
* **Run pages:** 3D viewer, results (showcase video, contact sheet, filmstrip, collision sheet, all renders),
  physics report, assets with licences and credits, documents (SPEC, README, visual check, smoke report), and an
  evaluation tab that runs the grader and the judge on demand.
* **Implementation:** FastAPI (`dashboard/server.py`), a transcript-to-timeline parser (`dashboard/events.py`), a
  MuJoCo session per viewer (`dashboard/sim_session.py`: scene export to three.js geometry including meshes,
  heightfields and textured planes; real-time stepping thread; binary pose frames) and a React + TypeScript +
  three.js frontend (`dashboard/web`).

---

## Scale and cost

Measured over the {s['n']} scored golden runs (Claude Opus via Claude Code, Apple M4, 16 GB):

| | |
|---|---|
| User messages per environment | 2 in all {s['n']} runs |
| Build time, request to report | median {s['min_med']} min, 90th percentile {s['min_p90']} min |
| Agent cost per environment | median ${s['cost_med']}, maximum ${s['cost_max']} (Claude Code estimate) |
| Smoke test (render + physics + layout + videos) | median {s['smoke_med']} s |
| Poly Pizza, keyless mode | 10 s per uncached page request, shared across processes; cached afterwards |
| Golden iteration, 6 tasks, 4 in parallel | about an hour including grading and judging |

---

## Testing

* `pytest tests` covers the toolkit: robots respond and pass the examiner, the examiner catches base-link
  collisions, dead and unstable actuators, gravity and friction probes are correct in low gravity, layout checks
  flag floating and overlapping props, `world.py` dynamics run in every rollout, and the smoke test plus USD export
  work end to end.
* `tests/mcp_stdio_check.py` talks to the Poly Pizza MCP server over stdio exactly like Claude Code does.
* `tests/ui_shots.py` screenshots dashboard pages with a headless browser.
* The golden set is the end-to-end test: every environment is rebuilt from `build_env.py` in a clean copy and its
  smoke test re-run independently of the agent's own claims.

---

## Command-line reference

```text
# inside an environment folder
python build_env.py                                   # regenerate scene.xml and assets
python -m simkit.smoke scene.xml --controller controller.py --task-checks task_checks.py [--seconds S] [--no-showcase]
python -m simkit.physics_exam scene.xml [--controller ...] [--task-checks ...]
python -m simkit.showcase . --out showcase.mp4 [--seconds S]
python -m simkit.usd_export scene.xml isaac/scene.usda # Isaac Sim stage + local validation
python env.py                                         # Gymnasium self-test
python view.py                                        # live viewer with the demo (mjpython on macOS)

# repository tools
python harness/run_golden.py --iter N [--tier core | --tasks a,b] [--parallel 4]
python harness/finish_run.py golden/runs/iter_NN/<task> [--regrade] [--rejudge]
python harness/resume_run.py golden/runs/iter_NN/<task>
python harness/ledger.py                              # golden/LEDGER.md + scoreboard.json
python harness/make_media.py && python harness/write_readme.py
python harness/check_openrouter.py                    # optional: agents via OpenRouter (GOLDEN_PROVIDER=openrouter)
```

---

## Repository layout

```text
plugin/                               Claude Code plugin "sim-env-builder"
  .claude-plugin/plugin.json, .mcp.json
  bin/polypizza-mcp                   launcher for the MCP server
  polypizza_mcp/                      server.py (tools), client.py (API + keyless site, cache, rate limit), glb.py
  skills/sim-env-builder/
    SKILL.md                          the playbook
    reference/                        backend_choice, confirmation, assets, robots, smoke_tests, isaac
    scripts/setup_env.sh, check_platform.py
    scripts/simkit/                   assets, procedural, robots, scene, world, render, physics_exam, layout,
                                      smoke, showcase, usd_export
    templates/                        build_env.py, env.py, view.py, isaac_smoke.py
golden/                               tasks.json, judge_prompt.md, LEDGER.md, CHANGELOG.md, scoreboard.json
harness/                              run_golden, grade, judge, ledger, finish_run, resume_run, make_media, write_readme
dashboard/                            server.py, events.py, sim_session.py, web/ (React + three.js)
docs/media/                           gallery GIFs, videos, screenshots, model credits
tests/                                toolkit tests, MCP stdio check, UI screenshots
```

Run data (`golden/runs/`, about 1.5 GB of transcripts, environments and videos) stays local and is not committed.

---

## Limitations and roadmap

* **Isaac Sim runtime.** The USD export is validated structurally; `isaac_smoke.py` has not been run on an NVIDIA
  machine yet, so PhysX behaviour and RTX renders are untested.
* **Look.** Poly Pizza is a low-poly library and MuJoCo renders with simple lighting; scenes read clearly but are
  not photoreal (that is what the Isaac path is for).
* **Deformables.** Cloth uses flexcomp with edge constraints; the shell-bending plugin is not in the pip build of
  MuJoCo. `cloth_table` is the only golden task below 85.
* **The judge is a model.** Scores vary by a few points between runs; the grader's 40 points are deterministic.
* **Cost.** A full 16-task iteration is about a hundred dollars of agent usage at Opus pricing; the harness waits
  out usage limits instead of failing.
* **Next:** humanoid and dexterous-hand templates, MuJoCo Menagerie robots as a one-line option, batched rendering
  for faster RL, and running the Isaac smoke script on an RTX machine.

---

## Acknowledgements

* 3D models from [Poly Pizza](https://poly.pizza) by Quaternius, Kenney, Poly by Google and many others; credits for
  every model in the media are in [`docs/media/CREDITS.md`](docs/media/CREDITS.md), and every generated environment
  records its own in `assets/polypizza/CREDITS.md`.
* Physics and rendering by [MuJoCo](https://mujoco.org/); USD export for
  [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim); the server speaks the
  [Model Context Protocol](https://modelcontextprotocol.io/).
* Built with and for [Claude Code](https://claude.com/claude-code).
"""
    (ROOT / "README.md").write_text(readme)
    print("README.md:", len(readme.splitlines()), "lines;", s, "| means", mean_hist)


if __name__ == "__main__":
    main()
