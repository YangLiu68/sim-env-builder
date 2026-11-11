# Robots, vehicles and task patterns (MuJoCo)

## Tested builders: `simkit.robots`

| Builder | Actuators (units) | Controller | Notes |
|---|---|---|---|
| `quadrotor(name, pos, mass=1)` | 4 x `motor` on sites: thrust N (0..2.5 mg/4), with yaw reaction torque | `QuadrotorController(model, name)`: `.target=(x,y,z)`, `.yaw` | FPV camera `<name>_fpv`; visual mesh via `visual_xml` |
| `car(name, pos, yaw_deg)` | `steer_l/r` position (rad, +-35 deg); `drive_l/r` motor (N*m, rear wheels); `brake_fl/fr/rl/rr` (set via `R.car_brake_torque`, never reverses) | `CarController(model, name, waypoints, speed)` pure pursuit | Cameras `<name>_driver`, `<name>_chase`; body origin at ground level |
| `diff_drive(name, pos)` | `left/right` velocity (rad/s) | `DiffDriveController(model, name, goal)` | `top_xml` puts a payload/arm on the deck |
| `arm(name, pos, dof=6, reach=0.85)` | `a1..aN` position (rad), `grip_l/r` position (0 open .. 0.04 closed) | `ArmController(model, name)`: `.target=xyz`, `.grip=0..1` (DLS IK, gripper down) | gravcomp links, wrist camera, EE site `<name>_ee` |
| `quadruped(name, pos)` | 12 position (rad): `<leg>_abd/hip/knee_act` | `StandController(model)`; `.gait=True` small trot | Home = standing pose; feet touch sensors |

`robot.add_to(scene)` adds body, actuators, sensors, excludes and registers the home pose.
Controllers are callables `ctrl(model, data)`; build them in `controller.py`:

```python
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from simkit import robots as R
def make_controller(model):
    c = R.QuadrotorController(model, "drone"); waypoints = [(3, 0, 2), (6, 0, 2), (9, 0, 2)]
    def ctrl(m, d):
        i = min(int(d.time // 3), len(waypoints) - 1); c.target = waypoints[i]; c(m, d)
    return ctrl
```
Scripted multi-phase tasks (pick and place, routes) = a small state machine on `data.time` or on
measured progress (distance to goal), as above.

## Tuning rules

- `integrator="implicitfast"`, timestep 0.002 (0.001 for stiff contact / small objects < 2 cm, 0.005 for big slow vehicles).
- Position actuators: `kp` such that `sqrt(kp / I_joint) * dt < 0.6`; `kv ~ 0.05-0.1 kp`; `forcerange` realistic.
- Every actuated joint: `armature` 0.01-0.5 and `damping`. Arm links: `gravcomp="1"` so the arm holds pose.
- Wheels: cylinders, `condim="3"`, friction ~1; exclude chassis<->wheel contacts.
- Grasping: finger pads `condim="4"` friction ~1.5; object mass realistic (mug 0.3 kg); timestep 0.002 or smaller.
- Visual mesh + simple collision: put the mesh in `group 2` (no contact, mass 0), the collider in `group 3`, and give the body an explicit `<inertial>` or massive collider.

## Category notes

**Self-driving / traffic**: `P.road(length, lanes)` strips at z=0 (surface 0.02 m), intersections from
crossing strips (put one 1 mm higher to avoid z-fighting), sidewalks as 0.15 m boxes, `traffic_light`,
`stop_sign`, Poly Pizza cars/buildings/trees. Ego = `R.car` (+ Poly Pizza body via `visual_xml`). NPC cars:
parked = static props; moving = `R.car` + `CarController` with their own waypoints, or kinematic mocap bodies
(`<body mocap="true">`, move `data.mocap_pos` each step: cheap, collision-free for the NPC itself).
Pedestrians: capsule people or Poly Pizza characters as mocap bodies walking along sidewalks/crosswalks.
Traffic light state: change lamp geom rgba via `model.geom_rgba` in the controller.

**Robot arms / tabletop**: `R.arm` on a table (base at table height), objects from Poly Pizza (mug, fruit)
or `P.mug/bottle/fruit/block`, free joints, resting on the table top. Task checks: reach accuracy,
grasp-and-lift (object z rises > 5 cm while gripper closed).

**Drones**: `R.quadrotor`; gates `P.gate`/`P.ring_gate`; wind via `<option wind>` or per-step `xfrc_applied`.
Task checks: hover holds +-5 cm, waypoint tracking, passes gate centres. Run controllability checks
through `env.step()` with action-space commands and a small impulse disturbance, not only through
the built-in cascaded controller. For RL racing/obstacle tasks put 1-2 obstacles within ~1 m of
the racing line (otherwise obstacles never shape the task) and give canopies a coarse collider.
Keep the FPV camera clear of the drone's own body (smoke hints when an onboard camera sees > 15%
of its own robot).

**Legged**: `R.quadruped` (or Menagerie Go2/ANYmal if the user wants a real robot), terrain via `P.terrain`
or `P.stairs`. Task checks: stands 3 s with trunk height within 5 cm, no trunk contact with ground,
all 4 foot touch sensors > 0 during the stand, feet start within 1 cm of the surface.
Stairs as a must-have get a PHYSICAL climbability probe, not just geometry: e.g. push a 0.3 m wheeled
test body or a foot-sized capsule up step 1 and assert it lands on it.
"Rough" terrain must look rough at robot scale: for a 0.35 m-tall quadruped use 8-15 cm bumps
(`P.terrain(kind="rough", max_height=0.15)` patches) away from the flat spawn pad; "rocky" means
several 0.3-1 m rocks near the route, not pebbles.

**Humanoid**: procedural humanoids are hard to balance; prefer MuJoCo Menagerie (unitree_h1/g1) after asking,
or the classic MuJoCo `humanoid.xml` structure; check standing with a PD hold for 2 s or a ragdoll fall test.

**Mobile manipulation / warehouse**: `R.diff_drive(top_xml=...)` with an arm body on the deck (or `R.arm`
placed on the deck body), `P.rack`, `P.pallet`, `P.crate`, conveyors (`P.conveyor`, move items by setting
their qvel each step or with a kinematic belt).

**Marine**: water as a translucent static box (`contype=0`), no collision. Floating (in `world.py`): each step apply
Archimedes force `rho * g * V_submerged` (+ linear/quadratic drag) via `data.xfrc_applied` on the hull body,
estimating submerged volume from hull draft (box hull: area * clamp(waterline - bottom_z, 0, h)).
Underwater vehicles: `<option density="1000" viscosity="0.0009"/>` gives buoyancy + drag for submerged bodies
(set body density/mass so it is neutrally buoyant).

**Space / planetary**: gravity (Moon 1.62, Mars 3.71, orbit 0), `P.terrain(kind="craters")`, regolith friction
0.6-0.8, rover = `R.car` scaled or 6-wheel variant (copy the wheel pattern).

**Multi-agent**: name every robot uniquely (`R.quadrotor("drone0")`, ...), one controller per agent,
env.py action = concatenation (document the order).

**Cloth / rope / soft**: `P.cloth_xml(name, size, n=11, pos=(x, y, support_top + 0.004))` gives a tested
flexcomp cloth (edge equality constraints, real-time at 9-15 vertices per side; the shell-bending plugin is
NOT available in the pip build). `P.cloth_vertex_bodies(model, name)` -> vertex body ids for grasping a corner
or checking drift. Rope: capsule chain with ball joints. Checks: passive drift over the stated duration,
corner lift height, no NaN; report the real-time factor (RL users care).

## Real robots from MuJoCo Menagerie (only if the user wants them)

`git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie && cd
mujoco_menagerie && git sparse-checkout set franka_emika_panda` then `<include file=".../panda.xml"/>` or
`spec.attach`. Check the licence in the robot folder; keep actuator names.
