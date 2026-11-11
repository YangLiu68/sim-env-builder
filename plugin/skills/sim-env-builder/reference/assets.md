# Assets: Poly Pizza first, procedural fallback

## Poly Pizza MCP tools (server `polypizza`)

| Tool | Use |
|---|---|
| `find_assets(needs=[{"role":"car","queries":["sedan","car"]}, ...])` | Batch search, one verdict per role. Start here. |
| `search_models(query, limit, licence)` | Single search with verdict + match quality. |
| `preview_models([ids])` | Labelled thumbnail grid (image). Look before downloading. |
| `get_model(id)` | Triangles, tags, category, licence, attribution. |
| `download_model(id, dest_dir, name)` | Saves `<name>.glb`, returns size/orientation facts, writes CREDITS.md + polypizza_assets.json. |
| `server_status()` | Mode (API key vs keyless), cache. |

Keyless mode respects poly.pizza's 10 s crawl delay: batch with `find_assets`, do not loop
`search_models` over dozens of queries. Results are cached for a week.

### Search tips
- Short concrete nouns: "forklift", "pallet", "office chair", "pine tree", "traffic cone".
- Try a synonym or a more generic word before falling back: "sedan" -> "car"; "cardboard box" -> "box".
- Verdicts: `found` = some title contains all query words; `weak` = partial matches only
  (e.g. "racing gate" matched "Fence Gate"); `not_found` = nothing relevant.
- Repeated categories (trees, rocks, cars, people): download 3-4 distinct models when good ones
  exist and vary scale/rotation, so the scene does not look copy-pasted.
- Prefer CC0 when two candidates are equal (no attribution needed). CC-BY is fine: credits are tracked.
- Reject: bundles of many objects in one file, wrong style (cartoon in realistic scene), very
  high triangle counts (>50k for a prop), animated/skinned characters if you need a static prop
  (skinned models import in T-pose).
- Objects that need live state (traffic-light lamps that switch, screens, doors that open, conveyors)
  are procedural from the start - say so in the brief instead of listing a static Poly Pizza model.
- Things Poly Pizza rarely has: articulated robots, drone racing gates, lab equipment, specific
  industrial machines, terrain, roads with markings -> procedural.

## Converting a GLB (simkit.assets.glb_to_mujoco)

```python
car = glb_to_mujoco("assets/polypizza/sedan.glb", ENV, "sedan", target={"length": 4.5}, yaw_deg=90, collision="box")
sc.add_asset(car); sc.body(car.body_xml("parked0", pos=(5, 3, 0), euler=(0, 0, 180)))
```
- `target`: one of length (x after yaw) / width / height / max / footprint, in metres, or
  `{"box": (L, W, H)}` to hit a planned size: uniform scale plus at most 20% per-axis correction;
  `asset.fit` reports wanted vs got and it warns when the model still misses by > 15%.
  Use "box" for furniture and anything with a planned footprint. Otherwise scaling is
  uniform on purpose. If a model's proportions are off by more than ~1.3x on some axis (a squat goal,
  a long-and-thin table), pick another candidate or go procedural instead of stretching it.
  Check feature scale inside composite models too: after sizing a fruit tree to 4 m, are its
  apples ~8 cm or 30 cm? If details come out wrong, pick another model or add details procedurally.
  Likewise avoid scaling a model more than ~4x its natural size (a 19 cm pebble as a 2 m boulder
  looks wrong: facet density and style give it away); find a model made for that size.
- `yaw_deg`: rotate so the model faces +x (vehicles, chairs). Check the closeup render.
- `collision`: `box` (cheap, default), `hull` (one convex hull), `parts` (hull per connected
  component - tables, shelves, arches, gates), `none` (decoration only).
- Origin is the footprint centre at the bottom: `pos=(x, y, support_height)` rests it on the support.
- For a dynamic prop: `body_xml(..., free=True, mass=kg)`.
- Asset `.size` gives the real extents; use it for spacing and stacking.

## Real-world sizes (use these unless the user says otherwise)

| Object | Size |
|---|---|
| Sedan / SUV / van / bus / truck | L 4.5 / 4.8 / 5.2 / 12 / 8-16 m, H 1.45 / 1.7 / 2.0 / 3.2 / 3.5 m |
| Person | 1.6-1.85 m tall |
| Door / storey height / ceiling | 2.0-2.1 m / 3.0-3.5 m / 2.5-2.8 m |
| Lane width / sidewalk / kerb | 3.5 m / 2-3 m / 0.15 m |
| Traffic light pole / street lamp / stop sign | 4.5-6 m / 6-9 m / 2.1 m |
| Tree (street / park / pine) | 4-8 m / 8-15 m / 10-25 m |
| Table / desk / kitchen counter height | 0.75 / 0.74 / 0.9 m |
| Chair seat / sofa seat | 0.45 / 0.42 m |
| Mug / can / bottle / apple | H 0.10 / 0.12 / 0.25 m, apple D 0.075 m |
| Pallet / warehouse rack / shelf unit | 1.2x1.0x0.144 m / 2.7x1.1x4-10 m / 1x0.4x2 m |
| Forklift | L 2.5-3 m, H 2.1 m (mast) |
| Small quadrotor / racing gate opening | 0.25-0.5 m span / 1.5-2 m |
| Quadruped (Go2/Spot class) | 0.3-0.6 m tall at the shoulder |

## Procedural fallback (simkit.procedural)

Recipes return a `Prop` (origin at footprint centre, bottom at z=0) with `body_xml(name, pos, euler,
free=False, mass=None)`; mesh-based ones need the env dir:
```python
P.tree(6, "pine"); P.building(12, 10, 24); P.house(); P.table(1.2, 0.8, 0.75); P.shelf(levels=5)
P.traffic_light(5); P.street_lamp(); P.road(60, lanes=4); P.gate(1.6, 1.6, 1.0); P.pallet(); P.crate()
P.mug(ENV, "mug0"); P.bottle(ENV, "b0"); P.rock(ENV, "rock3", radius=0.8, seed=3)
P.terrain("hills", size=(60, 60), max_height=4, kind="hills"|"rough"|"craters"|"dunes"|"steps", flat_center=5)
P.lathe(ENV, "vase", [(0,0),(0.1,0),(0.12,0.2),(0.06,0.35),(0,0.35)]); P.extrude(ENV, "sign", polygon, 0.02)
```
If no recipe fits, compose primitives with `P.geom(type, size, pos, color, ...)` into a `P.Prop`.
Match the colour palette and level of detail of the Poly Pizza models in the same scene.
Spawn on terrain with `P.hfield_height(model, "hills", x, y)` (build twice: compile, then place).

## Licences

`download_model` writes `assets/polypizza/CREDITS.md`; copy its lines into README. CC-BY needs
the attribution line wherever the environment (or renders of it) is distributed.
