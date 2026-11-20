# Water on land: flooded streets, fords, puddles, manhole fountains (`simkit.flood`)

Use it whenever vehicles or robots drive through standing or flowing water on the ground. Open
water with a floating hull (boats, buoys) is different: buoyancy + drag in `world.py` on the hull
(see the harbor pattern in robots.md); a flooded street is shallow water OVER a road.

## Build (build_env.py)

```python
from simkit import flood as F
spec = F.FloodSpec(
    regions=[(-30, -7, 30, 7), (-7, -30, 7, 30)],   # union of rectangles = where water stands (the roadway)
    depth=0.12,                                      # base depth (m)
    bands=[{"axis": "y", "lo": -7, "hi": -3.5, "depth": 0.38}],   # a deeper lane / gutter side
    pools=[{"center": [4, -5], "radius": 3.0, "depth": 0.10}],    # extra depth in a dip (gaussian)
    edge=1.5, flow=(0.0, -0.6), bed_z=ROAD_TOP_Z,    # bed_z = the road surface height under the water
    geysers=[F.Geyser("mh_corner", (5.5, -5.0), height=0.8, radius=1.2),
             F.Geyser("mh_mid", (-1.0, 2.5), height=0.5)])
F.storm_look(sc)                                     # INSTEAD of sc.sky(): grey sky, haze, fog
sc.ground(size=..., style="wet_asphalt")             # or keep your road boxes; wet look = reflectance
F.add_to_scene(sc, spec, env_dir=HERE, wakes=["ego", "npc1", "npc2"])
```
- Depth is measured from the flat road surface `bed_z`; only the water surface follows the depth
  field (lane-to-lane differences of 0.1-0.4 m are what crown, gutters and dips produce). The road
  stays flat for contact. Say this in SPEC.md.
- Put the deep band where the media/request shows it (gutter side, the dip at the intersection).
  Pick depths that make the task real: one lane clearly passable (< 0.2 m), one clearly not for a
  car (> 0.35 m), so "pick the shallowest passable lane" is a decision, not a coin toss.
- Visuals made for you: ONE texture baked for this spec over the whole flood
  (`assets/procedural/flood_water.png`): streaks that follow the current and bend around each
  fountain's outflow, darker deep water, ragged foam boiling out of every manhole and trailing
  downstream, broken foam at the waterline. Per fountain a low lumpy foam mound that pulses with
  the jet plus blobs churning at the surface (mocap, animated); a bow wave ahead of each vehicle
  named in `wakes` (shows only when it moves through water). Water and foam are skipped by the
  layout checks. Set `spec.flow` and the geysers where the media shows them and the foam lands in
  the right places; do not paint your own foam discs or white ellipsoids on top.
- Wet grip needs the tyre friction to be THE contact friction: `robots.car` tyres carry
  `priority="1"`. A hand-made vehicle must do the same (MuJoCo otherwise uses the max of tyre and
  road friction and lowering the tyre's does nothing). `robots.car(..., wheelbase=, track=)` puts
  the physics wheels under a mesh's wheels; `extra_xml=` adds a sensor dome/lidar/cameras to the body.
- `storm_look(sc, fog_start=0.25, fog_end=1.6)`: fog distances are fractions of the scene extent;
  lower them if the media shows distant trees fading into mist. simkit renders show fog (smoke,
  showcase, env.py camera, reference compare).

## World (world.py)

```python
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from simkit import flood as F
def make_world(model):
    flood = F.FloodWorld(model)                      # spec from scene.xml; every simkit car is a vehicle
    lights, npcs = ..., ...                          # traffic lights, NPC drivers (CarController)
    def world(model, data):
        flood(model, data)                           # SETS xfrc_applied of each vehicle: add other forces after it
        lights(model, data); npcs(model, data)
    return world
```
Physics per vehicle (all in `flood.py` docstring): quarter-footprint buoyancy (fill 0.65, underside
0.22 m: a 2 t car loses ~90 % of its tyre load at 0.6 m), drag `0.5 rho Cd A v_rel^2` on the immersed
frontal area (wheels below the underside at 35 % blockage, full width above), tyre friction
`mu_dry*(1-0.35*min(d/0.3,1))`, the current pushing through the drag term, fountain jets
`rho*A_open*2*g*h` pushing up at the manhole. `FloodWorld.info[name]` = depth, buoyancy, drag, jet now.
Per-vehicle overrides: `F.FloodWorld(model, vehicles={"ego": {"clearance": 0.18}, "npc1": None})`.

## Driving logic that uses the water

- Lane choice: sample `spec.depth_at(x, y)` along each candidate lane centre line over the flooded
  stretch; passable = max depth < the vehicle's safe wading depth (sedans ~0.25 m, SUVs/robotaxis
  ~0.3-0.45 m; state the number in SPEC.md); choose the shallowest passable lane, waypoints along it.
- Do not stop inside the flood: the controller keeps a minimum speed (2-3 m/s) until the ego has
  left the wet region (`spec.depth_at(ego) < 0.01`); stops (lights, queues) happen before or after.
- Fountains on the route: drive PAST them, do not stop in front of them in the water (the failure
  the robotaxi in the SF flood videos showed). The car plans from what it can see: fountains that
  are erupting now (`gz.active(t)`), not the whole spec. Every step, for an erupting fountain within
  ~35 m ahead:
  ```python
  plan = F.pass_fountain(spec, gz, (x, lane_y), (1, 0), half_width=1.03, safe_depth=SAFE,
                         lateral_limits=(-ROAD_HALF + 0.3 - lane_y, ROAD_HALF - 0.3 - lane_y),
                         obstacles=[(px, py, 2.6) for parked cars],
                         movers=[(x, y, vx, vy, 3.0) for moving cars], speed=V_PASS, margin=1.0)
  if plan and plan["offset"] is not None:     # the passable side: depth under the whole car < SAFE,
      ctl.waypoints = F.swerve_path((x, lane_y), (1, 0), plan, 3.0, goal_s)   # inside kerbs, free
  elif plan:                                  # neither side passable: stop BEFORE the water
      ...
  ```
  `plan["candidates"]` says why a side was rejected (road edge / too deep / obstacle); log it.
- Reactive scenarios: `F.Geyser(..., start=9.0)` erupts at t = 9 s (dormant and invisible before,
  no baked foam): time it from a dry run so the ego is 20-25 m away, and compare a "stop and wait"
  baseline with the detour (time stopped in water, min speed, clearance, max depth, collisions).
- Slow down before deep water (bow wave, drag); wakes look right at 2-5 m/s.

## Task checks (measured physics, not read-backs of the forces you set)

```python
from simkit import flood as F
spec = F.FloodSpec.from_model(model)
def check_drag_grows_with_depth(model, data):
    w = F.FloodWorld(model)
    xs, ys, ds = F.find_spot(spec, 0.10); xd, yd, dd = F.find_spot(spec, 0.35)
    a = F.coast_test(model, w, "ego", (xs, ys), yaw_deg=0, v0=5.0)
    b = F.coast_test(model, w, "ego", (xd, yd), yaw_deg=0, v0=5.0)
    ok = b["decel"] > 1.5 * a["decel"] > 0
    return {"status": "pass" if ok else "fail", "value": {"shallow": a, "deep": b, "depths": [ds, dd]},
            "expect": "deep-water deceleration > 1.5x shallow", "details": "coasting at 5 m/s, zero throttle"}
def check_fountain_pushes_up(model, data):
    r = F.fountain_uplift(model, "ego", "mh_corner")   # tyre load with the jet on vs off, from contacts
    ok = r["uplift_N"] > 0.5 * r["jet_force_predicted_N"] > 0
    ...
```
- Max depth the ego entered: record `spec.depth_at` at the ego's wheels every step of the demo
  rollout (and 2-3 perturbed ones), compare with the safe wading depth.
- "Does not stop in the flood": min speed while `depth_at(ego) > 0.02` over the rollout.
- "Passes a fountain instead of stopping": with a `start`-delayed geyser on the ego's lane, the
  rollout has 0 s stopped in water, clearance from the car body to the fountain radius > 0.3 m,
  max depth under the tyres < safe depth, no contacts with other cars.
- Choose coast-test spots away from other cars and fountains (`find_spot(..., near=(x, y))`) and
  aim the heading along the street so the car stays in that depth band.
