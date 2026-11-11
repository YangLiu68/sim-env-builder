# onellmpj: sim-env-builder

A Claude Code **skill + MCP server** that builds 3D simulated environments (MuJoCo or NVIDIA Isaac Sim)
from a description, with a **golden set** and a self-improvement loop that tests it with fresh agents,
and **SimForge Studio**, a dashboard to run it, watch it and drive the results in 3D.

```
plugin/                          Claude Code plugin "sim-env-builder"
  .claude-plugin/plugin.json
  .mcp.json                      registers the Poly Pizza MCP server
  bin/polypizza-mcp              launcher (finds a Python with mcp/trimesh/Pillow)
  polypizza_mcp/                 MCP server: find_assets, search_models, preview_models, get_model, download_model, server_status
  skills/sim-env-builder/
    SKILL.md                     the playbook (phases: setup, backend choice, confirm, assets, build, smoke tests, Isaac, deliver)
    reference/                   backend_choice, confirmation, assets, robots, smoke_tests, isaac
    scripts/setup_env.sh         shared sim venv (~/.cache/sim-env-builder/venv)
    scripts/check_platform.py    GPU / OS / Isaac facts for the backend decision
    scripts/simkit/              helper library copied into every environment
      assets.py                  GLB -> MuJoCo (real-world size, Z-up, colour parts, box/hull/parts colliders), GLB -> USD
      procedural.py              fallback props (60+ recipes), lathe/extrude/rock meshes, heightfield terrain
      robots.py                  tested quadrotor, car, diff-drive, arm+gripper, quadruped + controllers (IK, pure pursuit...)
      scene.py                   MJCF builder: sky, ground, look-at cameras, name-based home keyframe
      render.py                  auto views, contact sheets, filmstrips, blank-frame detection, segmentation visibility
      physics_exam.py            physics examiner (solver probes, rollout, actuators, determinism, ...)
      layout.py                  floating / sunken / interpenetrating props
      smoke.py                   one command for render + physics + layout + filmstrip + report
      usd_export.py              MuJoCo -> USD with UsdPhysics for Isaac Sim (+ local validation)
    templates/                   build_env.py, env.py (Gymnasium API), isaac_smoke.py
golden/
  tasks.json                     16 golden tasks across categories (prompt + simulated user reply + must-haves)
  judge_prompt.md                rubric for the independent judge
  runs/iter_NN/<task>/           transcripts, agent replies, grade.json, judge.json, regrade renders, env
  scoreboard.json, LEDGER.md     aggregated scores;  CHANGELOG.md: what each iteration changed
harness/
  run_golden.py                  fresh headless agents per task: shot 1 (confirm) -> shot 2 (build) -> grade -> judge
  grade.py                       automated 40 pts: deliverables, confirmation, clean rebuild, independent smoke re-run, env.py, visual-check evidence, assets
  judge.py                       independent claude -p judge, 60 pts, looks at the renders
  ledger.py                      scoreboard + LEDGER.md
dashboard/                       SimForge Studio (FastAPI + React/three.js)
tests/                           simkit regression tests, MCP stdio check, UI screenshot script
```

## Results so far

Golden-set mean score per iteration (automated 40 + independent judge 60): iter_02: 88.7, iter_03: 89.8, iter_04: 91.0, iter_05: 90.8.
Every scored run confirmed its plan before building; nearly all finished in 2 user messages.

| task | category | latest score | iteration |
|---|---|---|---|
| warehouse_amr | logistics | 94.0 | iter_02 |
| orchard_drone | agriculture | 94.0 | iter_04 |
| drone_gates | aerial | 93.5 | iter_04 |
| factory_sorting | industrial | 93.0 | iter_03 |
| home_fetch | mobile-manipulation | 92.5 | iter_04 |
| arm_kitchen | robot-arm | 91.5 | iter_04 |
| isaac_synthetic | isaac-preferred | 91.5 | iter_05 |
| sd_intersection | self-driving | 91.5 | iter_05 |
| cartpole_minimal | classic-control | 91.0 | iter_04 |
| quadruped_trail | legged | 90.5 | iter_05 |
| humanoid_gym | humanoid | 90.5 | iter_03 |
| mars_rover | space | 90.5 | iter_05 |
| robot_soccer | multi-agent | 90.5 | iter_03 |
| harbor_boat | marine | 90.0 | iter_05 |
| lab_bench | fallback-stress | 89.5 | iter_03 |
| cloth_table | deformable | 83.4 | iter_04 |

Full history: `golden/LEDGER.md`; what each round changed: `golden/CHANGELOG.md`.
Run data (`golden/runs/`, ~1.5 GB of transcripts, environments and videos) stays local and is not committed.

## Use the skill

```bash
claude --plugin-dir plugin            # or install the plugin; then just ask:
> Build me a warehouse simulation with racks, pallets and a mobile robot
```
The agent checks the machine, picks the backend, sends a short plan with 2-4 questions, then builds
`./<env>/` (scene.xml, build_env.py, env.py, controller.py, task_checks.py, assets, smoke report) and
smoke-tests it: renders, a physics examiner, and its own look at the images.

Poly Pizza: works without a key (reads the public site politely, 10 s between page requests, cached);
set `POLY_PIZZA_API_KEY` (free at https://poly.pizza/settings/api) for the official API.

## Dashboard

```bash
.venv/bin/python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8766
open http://127.0.0.1:8766
```
- **Studio**: describe an environment, watch the agent work live (Poly Pizza searches with verdicts and
  preview grids, build steps, smoke tests), answer its confirmation questions, then open the result.
- **3D viewer**: the MuJoCo scene runs on the server in real time and streams to three.js; actuator
  sliders, demo controller toggle, named/robot cameras, follow mode, collision overlay, Shift+click push.
- **Golden set**: score per task per iteration, trend, judge findings.
- Frontend dev: `cd dashboard/web && npm install && npm run dev` (proxies /api to :8766); `npm run build` for the served bundle.

## Golden loop

```bash
.venv/bin/python harness/run_golden.py --iter 2 --tier core --parallel 4     # or --tasks a,b
.venv/bin/python harness/ledger.py                                            # refresh scoreboard
.venv/bin/python -m pytest tests -q                                           # toolkit regression tests
```
Score = automated (40) + judge (60). Fix what the judge and the grader flag in the plugin, record it in
`golden/CHANGELOG.md`, run the next iteration.
