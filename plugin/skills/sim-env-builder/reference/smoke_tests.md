# Smoke tests: what each check means and how to fix it

Run: `$PY -m simkit.smoke scene.xml --out smoke --controller controller.py --task-checks task_checks.py`
(options: `--seconds`, `--size 1280x720`, `--allow-floating lamp,sign`, `--expected-sizes sizes.json`
with `{"tree0": [4, 8]}` heights in metres). Physics only: `$PY -m simkit.physics_exam scene.xml ...`.

## Render test (render_report.json)

| Problem | Meaning | Fix |
|---|---|---|
| blank / almost one colour | camera inside geometry, pointing at sky/void, or scene invisible | move camera (`Scene.camera(pos, target)`), check geom groups 0-2 are visual |
| too dark / washed out | lights | `sc.sky("day")` adds sun + headlight; avoid extra bright lights |
| no detail | camera too far or object too small | add a closer camera |
| never visible objects | top-level body hidden in every view | it is buried, too small, outside the scene, or rgba alpha 0 |

## Physics examiner (physics_report.json)

| Check | Fails when | Usual fix |
|---|---|---|
| compile | XML invalid | message has the line |
| solver_options | dt > 10 ms | smaller timestep |
| mass_inertia | moving body without mass; mass ratio > 200 across a joint | give colliders mass / explicit `<inertial>`; heavier small links |
| actuator_config | omega*dt > 0.6, zero gear | lower kp / add armature / smaller dt |
| initial_penetration | geoms overlap at reset (> 5 mm warn, > 2 cm fail) | move objects; add `<exclude>` for robot base vs first link; collision geoms smaller than visuals |
| numerical_stability | NaN or |qvel| > 100 | almost always gains, masses or initial penetration |
| solver_warnings | contact buffer full, bad qacc | fewer contacts (simpler colliders), `<size nconmax>` ... |
| max_penetration | contacts sink > 1 cm | mass too high for solref; use `solref="0.005 1"` on heavy objects; smaller dt |
| fall_through | free body below the ground / flew away | spawn above support; thin ground boxes -> use plane |
| passive_settling | unactuated free objects moved > 20 cm or still moving | spawned in the air or overlapping; place exactly on support |
| joint_limits | joints overshoot range | stronger limits (`solreflimit`), lower gains |
| freefall_gravity | probe sphere does not follow g | gravity option wrong |
| resting_contact | box sinks > 5 mm or jitters | ground `solref`/`solimp` too soft; dt too large |
| coulomb_friction | sliding deceleration differs from mu*g by > 15% | friction attribute wrong, condim 1 |
| bounce_energy | contacts inject energy | solref damping ratio < 1 |
| box_stack | 3 boxes do not stay stacked | solver iterations, dt, friction |
| buoyancy | (fluid scenes) light objects sink / heavy float | density/viscosity option |
| determinism | two runs differ | randomness in controller without seed |
| timestep_sensitivity | dt vs dt/2 differ > 5 cm | chaotic contact (ok) or under-resolved dynamics |
| realtime_factor | < 1x | simplify meshes/colliders |
| actuator_response | actuator does not move its joint (dead) or moves the wrong way | gear 0, joint blocked by contact/limit, too weak (forcerange), sign of gear |
| floating_objects | static prop with a gap below | set z to the support height (prop origin is its bottom) |
| sunken_objects | prop bottom > 5 cm under ground | same |
| static_overlaps | two static props interpenetrate | spacing; use `asset.size` |
| task checks | your criteria | fix the env or the controller, never weaken the check to pass |

## Visual check (you, looking at the images)

Open with the Read tool: `smoke/contact_sheet.png`, `smoke/filmstrip.png`, `smoke/collision_sheet.png`,
then the closeups in `smoke/renders/` (`auto_robot_*`, named cameras). Write `smoke/visual_check.md`:

```
| # | Question | Answer |
| 1 | Every requested object visible + recognisable? | yes - 4 cars, 2 lights, ... / no - X missing |
| 2 | Real-world scale right? | ... |
| 3 | Upright, facing right, resting on supports? | ... |
| 4 | Poly Pizza models correct (no black/missing parts)? | ... |
| 5 | Procedural fallbacks read as intended objects? | ... |
| 6 | Lighting/sky/ground fine in all views? | ... |
| 7 | Filmstrip motion as intended? | ... |
| 8 | Colliders match visible shapes? | ... |
| 9 | Matches the user's request? Missing? | ... |
Verdict: PASS / FAIL (+ what you changed after looking)
```
Common visual bugs: model facing sideways (fix yaw_deg), giant/tiny models (fix target size),
objects floating 1 m up (wrong z), cars sunk into road (road surface is z=0.02), black meshes
(material without colour - check `asset.visual` rgba), camera inside a wall.
