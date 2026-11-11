# Confirmation brief (send once, before building)

Keep it under ~25 lines. The user should be able to answer with "yes" or a few words.
Every question carries your recommended default, so "yes, go" is always a complete answer.

## Template

```
Plan for <env name> (<one-line summary of the request>)

Backend: MuJoCo <version> - <one-line reason>. [Isaac: <why not / or offered as USD export>]
Scene: <bullet list of zones + objects with counts and real sizes, e.g. "4-way intersection,
  2x2 lanes, 3.5 m lanes; 6 buildings 10-30 m; 2 traffic lights; 8 trees">
Agent: <robot/vehicle, DoF, actuators (units), sensors/cameras, control rate>
Assets: Poly Pizza -> <roles found> | procedural -> <roles not found / rejected>
Physics: gravity <g>, ground friction <mu>, timestep <dt> s, <special: wind, fluid, low gravity>
Demo + tests: scripted <controller> ; task checks: <2-4 checks> ; full smoke tests (render,
  visual review, physics examiner)
Deliverables: ./<env>/ scene.xml, build_env.py, env.py (Gym API), README, smoke report + images

Questions (defaults in brackets):
1. <most important ambiguity>? [<default>]
2. ... [<default>]
Reply "go" to build with the defaults.
```

## What to ask about (pick the 2-4 that matter most for THIS request)

- Backend, only if genuinely ambiguous or Isaac is preferred but not runnable here.
- Robot model: procedural (simkit.robots, instant, tested) vs a specific real robot
  (Franka, UR5e, Unitree Go2, Skydio X2... from MuJoCo Menagerie, needs a download).
- Scale/extent: how big, how many objects/agents.
- Control interface: torque vs position vs velocity; continuous vs discrete actions.
- Observations: state vector vs camera images (which cameras, resolution).
- Task: what success means (for reward stub + task checks).
- Visual style: realistic vs stylised low-poly (Poly Pizza is low-poly).
Do NOT ask about things with an obvious default (lighting, timestep, file names), and do NOT
re-ask anything the request already settles (if they said "thrust per rotor", don't ask about the
action space). Every question must be a real ambiguity in THIS request.
"Standard"/"classic" means the reference values exactly (Gym CartPole g=9.8, masses, lengths);
state any deviation in the brief, not after building.
Defaults are the SIMPLEST design that meets the request (skid-steer 4-wheel rover, conveyor that
sets box velocities, built-in simkit robot); offer the fancier version (rocker-bogie, friction
belt, Menagerie robot) as the alternative in the question, not as the default.
Size and counts too: default to the smallest scene that meets every must-have (4 tree rows, one-row
demo, 2 aisles); offer bigger as the alternative.

## SPEC.md template

```
# <env name>
Request: "<verbatim user request>"
Confirmed: <date>, user answers: <verbatim or summary>
Backend: <MuJoCo/Isaac> - <reason>
## Contents
<objects with counts, sizes, positions/zones>
## Agent
<robot, actuators with units/ranges, sensors, cameras, controller>
## Physics
<gravity, friction, timestep, integrator, special effects>
## Success criteria
<task checks + smoke test expectations>
## Assumptions
<anything decided without the user>
```
