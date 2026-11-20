"""Regression tests for the skill's simkit toolkit (run: .venv/bin/python -m pytest tests -q)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugin/skills/sim-env-builder/scripts"
sys.path.insert(0, str(SCRIPTS))

from simkit import physics_exam as PE  # noqa: E402
from simkit import procedural as P  # noqa: E402
from simkit.scene import Scene  # noqa: E402


def arm_scene(d: Path, exclude=True):
    sc = Scene("arm"); sc.sky("studio"); sc.ground(10, "tiles")
    t = P.terrain("bumps", size=(6, 6), max_height=0.3, kind="rough", seed=1, res=48)
    sc.add_asset(t); sc.body(f'<body name="terrain" pos="5 0 0">{t.geoms_xml()}</body>')
    sc.body(P.table().body_xml("table"))
    sc.body('''<body name="base" pos="0 0 0.75"><geom type="cylinder" size="0.06 0.05" pos="0 0 0.05"/>
      <body name="l1" pos="0 0 0.1"><joint name="j1" axis="0 0 1" range="-170 170" damping="1" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04"/>
        <body name="l2" pos="0 0 0.3"><joint name="j2" axis="0 1 0" range="-120 120" damping="1" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.035"/></body></body></body>''')
    sc.body(P.block().body_xml("cube", pos=(0.3, 0.1, 0.75), free=True))
    sc.actuator('<position name="a1" joint="j1" kp="50" kv="5" ctrlrange="-2.9 2.9"/>')
    sc.actuator('<position name="a2" joint="j2" kp="50" kv="5" ctrlrange="-2 2"/>')
    if exclude:
        sc.contact('<exclude body1="base" body2="l1"/>')
    sc.camera("front", (1.5, -1.5, 1.4), (0, 0, 0.9))
    return sc.write(d / "arm.xml")


def by_name(rep):
    return {c["name"]: c for c in rep["checks"]}


def test_arm_passes_and_actuators_respond(tmp_path):
    rep = PE.examine(arm_scene(tmp_path), seconds=1.5)
    c = by_name(rep)
    assert c["actuator_response"]["status"] == "pass", c["actuator_response"]
    assert c["actuator_response"]["value"]["responsive"] == 2
    assert rep["summary"]["fail"] == 0, [x for x in rep["checks"] if x["status"] == "fail"]


def test_examiner_catches_base_link_collision(tmp_path):
    rep = PE.examine(arm_scene(tmp_path, exclude=False), seconds=1.0)
    assert by_name(rep)["initial_penetration"]["status"] == "fail"


def test_examiner_catches_dead_and_unstable(tmp_path):
    xml = tmp_path / "bad.xml"
    xml.write_text('''<mujoco><option timestep="0.01"/><worldbody><geom type="plane" size="5 5 .1"/>
      <body name="arm" pos="0 0 1"><joint name="j" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="0.01"/></body>
      <body name="slider" pos="1 0 .5"><joint name="s" type="slide" axis="1 0 0" range="0 0.0001"/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody><actuator><position joint="j" kp="5000" ctrlrange="-1 1"/><motor joint="s" gear="0.0000001" ctrlrange="-1 1"/></actuator></mujoco>''')
    c = by_name(PE.examine(xml, seconds=1.0))
    assert c["actuator_config"]["status"] in ("warn", "fail")  # stiff gain vs dt
    assert c["actuator_response"]["status"] == "fail"           # slider is dead


def test_probes_detect_gravity_and_friction(tmp_path):
    sc = Scene("moon", gravity=(0, 0, -1.62)); sc.ground(20, "moon", friction="0.6 0.005 0.0001")
    c = by_name(PE.examine(sc.write(tmp_path / "moon.xml"), seconds=1.0))
    assert c["freefall_gravity"]["status"] == "pass"
    assert abs(c["coulomb_friction"]["value"]["mu_g"] - 0.6 * 1.62) < 1e-3
    assert c["coulomb_friction"]["status"] == "pass"


def test_layout_flags_floating_and_overlap(tmp_path):
    sc = Scene("lay"); sc.ground(20)
    sc.body(P.table().body_xml("table_ok"))
    sc.body(P.table().body_xml("table_floating", pos=(3, 0, 0.5)))
    sc.body(P.crate().body_xml("crate_a", pos=(-3, 0, 0)))
    sc.body(P.crate().body_xml("crate_b", pos=(-3.05, 0, 0)))
    from simkit.layout import check_layout
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(sc.write(tmp_path / "lay.xml")))
    c = {x["name"]: x for x in check_layout(m, mujoco.MjData(m))}
    assert any("table_floating" in s for s in c["floating_objects"]["value"])
    assert not any("table_ok" in s for s in c["floating_objects"]["value"])
    assert c["static_overlaps"]["status"] == "warn"


def test_smoke_and_usd_export(tmp_path):
    xml = arm_scene(tmp_path)
    from simkit import smoke, usd_export
    rep = smoke.run(xml, tmp_path / "smoke", seconds=1.0, size=(480, 270))
    assert rep["verdict"] in ("PASS", "PASS_WITH_WARNINGS"), rep
    assert (tmp_path / "smoke/contact_sheet.png").exists() and (tmp_path / "smoke/filmstrip.png").exists()
    info = usd_export.export(xml, tmp_path / "isaac/scene.usda")
    assert info["joints"] == 2 and info["drives"] == 2
    v = usd_export.validate(tmp_path / "isaac/scene.usda", xml)
    assert v["ok"], v["problems"]


def test_world_dynamics_run_in_examiner(tmp_path):
    """world.py forces (here: buoyancy) must apply in every examiner rollout."""
    sc = Scene("float"); sc.ground(20)
    sc.body('<body name="buoy" pos="0 0 2"><freejoint/><geom type="box" size=".3 .3 .3" mass="40"/></body>')
    xml = sc.write(tmp_path / "scene.xml")
    rep0 = PE.examine(xml, seconds=2.0)
    (tmp_path / "world.py").write_text(
        "import numpy as np, mujoco\n"
        "def make_world(model):\n"
        "    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'buoy')\n"
        "    def world(m, d):\n"
        "        depth = np.clip(2.0 - (d.xpos[b][2] - 0.3), 0, 0.6)\n"
        "        d.xfrc_applied[b, 2] = 1000 * 9.81 * 0.36 * depth - 200 * d.cvel[b][5]\n"
        "    return world\n")
    rep1 = PE.examine(xml, seconds=4.0)
    z0 = by_name(rep0)["passive_settling"]["status"] if "passive_settling" in by_name(rep0) else None
    assert z0 == "warn"  # without water it falls 1.7 m to the ground
    assert by_name(rep1)["passive_settling"]["status"] == "pass", by_name(rep1)["passive_settling"]  # floats at the waterline


# ----------------------------------------------------------------------------- flood + reference media
def flood_scene(d: Path):
    from simkit import flood as F
    from simkit import robots as R

    sc = Scene("flood")
    F.storm_look(sc)
    sc.ground(size=60, style="wet_asphalt")
    spec = F.FloodSpec(regions=[(-25, -8, 25, 8)], depth=0.10, bands=[{"axis": "y", "lo": -8, "hi": -3, "depth": 0.35}],
                       flow=(0.0, -0.5), geysers=[F.Geyser("mh1", (0.0, 2.0), height=0.8)])
    F.add_to_scene(sc, spec, env_dir=d, wakes=["ego"])
    R.car("ego", pos=(-30, 0, 0), mass=2100).add_to(sc)
    sc.camera("overview", pos=(-14, -20, 9), target=(0, 0, 0))
    return sc.write(d / "scene.xml")


def test_flood_spec_roundtrip_and_depth_field(tmp_path):
    import mujoco
    from simkit import flood as F

    m = mujoco.MjModel.from_xml_path(str(flood_scene(tmp_path)))
    sp = F.FloodSpec.from_model(m)
    assert sp.geysers[0].name == "mh1" and F.find_vehicles(m) == ["ego"]
    assert abs(sp.depth_at(0, 0) - 0.10) < 1e-6 and abs(sp.depth_at(0, -5.5) - 0.35) < 1e-6
    assert sp.depth_at(0, 7.95) < 0.01 and sp.depth_at(0, 20) == 0.0      # shoreline taper, dry outside
    assert (tmp_path / "assets/procedural/flood_water.png").exists()


def test_flood_drag_grows_with_depth_and_dry_is_unaffected(tmp_path):
    import mujoco
    from simkit import flood as F

    m = mujoco.MjModel.from_xml_path(str(flood_scene(tmp_path)))
    w = F.FloodWorld(m)
    dry = F.coast_test(m, w, "ego", (-35, 0), 0.0, v0=5.0)
    shallow = F.coast_test(m, w, "ego", (-12, 0.5), 0.0, v0=5.0)
    deep = F.coast_test(m, w, "ego", (-12, -5.5), 0.0, v0=5.0)
    assert dry["decel"] < 0.15 < shallow["decel"] and deep["decel"] > 2 * shallow["decel"]


def test_flood_buoyancy_unloads_tyres_and_fountain_pushes_up(tmp_path):
    import mujoco
    from simkit import flood as F

    m = mujoco.MjModel.from_xml_path(str(flood_scene(tmp_path)))
    weight = m.body_subtreemass[m.body("ego").id] * 9.81
    loads = {}
    for depth in (0.15, 0.6):
        w = F.FloodWorld(m, F.FloodSpec(regions=[(-100, -100, 100, 100)], depth=depth, edge=0.01), grip=False)
        d = F._fresh(m)
        F.place_vehicle(m, d, "ego", 0, 0, 0)
        for _ in range(1500):
            w(m, d)
            mujoco.mj_step(m, d)
        loads[depth] = F.wheel_load(m, d, "ego") / weight
    assert loads[0.15] > 0.97 and loads[0.6] < 0.2          # cars float at ~0.6 m of water
    up = F.fountain_uplift(m, "ego", "mh1")
    assert up["uplift_N"] > 0.8 * up["jet_force_predicted_N"] > 500


def test_flood_world_is_reset_safe_and_keeps_dry_friction(tmp_path):
    import mujoco
    from simkit import flood as F

    m = mujoco.MjModel.from_xml_path(str(flood_scene(tmp_path)))
    tire = m.geom("ego_fl_tire").id
    mu = float(m.geom_friction[tire, 0])
    for _ in range(3):  # several worlds on one model (examiner rollouts) must not compound the wet-grip factor
        w = F.FloodWorld(m)
        d = F._fresh(m)
        F.place_vehicle(m, d, "ego", 0, -5.5, 0)
        w(m, d)
    assert abs(m.geom_friction[tire, 0] - mu * (1 - 0.35)) < 1e-6
    # the CONTACT uses the wet tyre friction (tyres have priority; MuJoCo would otherwise take the
    # max with the ground's 1.0 and the water would not make the road slippery at all)
    for _ in range(50):
        w(m, d)
        mujoco.mj_step(m, d)
    tyre_mu = [d.contact[i].friction[0] for i in range(d.ncon) if tire in (d.contact[i].geom1, d.contact[i].geom2)]
    assert tyre_mu and max(tyre_mu) < 0.8


def test_flood_delayed_eruption_and_pass_fountain():
    import math

    import numpy as np
    from simkit import flood as F

    sp = F.FloodSpec(regions=[(-40, -9, 30, 9)], depth=0.26, blend=1.2,
                     bands=[{"axis": "y", "lo": 3.4, "hi": 20, "depth": 0.14}, {"axis": "y", "lo": -60, "hi": -3.6, "depth": 0.42}],
                     geysers=[F.Geyser("mh", (-2.0, 5.1), radius=1.1, start=9.0)])
    g = sp.geysers[0]
    assert g.strength(8.9) == 0 and not g.active(8.9) and g.active(9.5) and g.jet_force(5.0) == 0
    assert np.allclose(sp.flow_at(-2.0, 7.0, 5.0), 0)                    # dormant: no outflow either
    kw = dict(half_width=1.03, safe_depth=0.3, lateral_limits=(-9 + 0.3 - 4.95, 9 - 0.3 - 4.95))
    route = ((-30.0, 4.95), (1.0, 0.0))
    pl = F.pass_fountain(sp, g, *route, **kw)
    assert pl["side"] == "right" and pl["candidates"][0]["blocked"] == "road edge" and pl["max_depth"] < 0.3
    path = F.swerve_path(*route, pl, 0.0, 60.0, 1.0)
    assert min(math.hypot(x + 2.0, y - 5.1) for x, y in path) >= pl["keep"] - 0.05
    assert path[0][1] == 4.95 and path[-1][1] == 4.95                    # back in the lane after it
    assert F.pass_fountain(sp, g, *route, **{**kw, "safe_depth": 0.2})["offset"] is None   # too deep: stop
    leaving = [(-15.0, 1.5, 0.0, -4.0, 3.0)]                               # gone before we get there
    staying = [(-2.0, 2.0, 0.0, 0.0, 3.0)]                                 # stopped right where we would pass
    assert F.pass_fountain(sp, g, *route, movers=leaving, speed=3.5, **kw)["offset"] is not None
    assert F.pass_fountain(sp, g, *route, movers=staying, speed=3.5, **kw)["offset"] is None
    assert F.pass_fountain(sp, g, (-30.0, -1.65), (1.0, 0.0), **kw) is None   # not in the way


def test_flood_scene_passes_smoke_layout_and_reference_compare(tmp_path):
    import json as _json

    from PIL import Image
    from simkit import smoke

    scene = flood_scene(tmp_path)
    (tmp_path / "world.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, %r)\n"
                                       "from simkit import flood as F\n\ndef make_world(model):\n    return F.FloodWorld(model)\n" % str(SCRIPTS))
    (tmp_path / "reference/frames").mkdir(parents=True)
    Image.new("RGB", (360, 640), (120, 110, 100)).save(tmp_path / "reference/frames/ref.jpg")
    (tmp_path / "reference/match.json").write_text(_json.dumps([{"camera": "overview", "frame": "reference/frames/ref.jpg"}]))
    rep = smoke.run(scene, tmp_path / "smoke", seconds=2.0, showcase=False)
    assert rep["verdict"] in ("PASS", "PASS_WITH_WARNINGS"), rep["failed_checks"]
    phys = _json.loads((tmp_path / "smoke/physics_report.json").read_text())
    flagged = [c for c in phys["checks"] if c["group"] == "layout" and c["status"] != "pass" and "flood" in _json.dumps(c["value"])]
    assert not flagged, flagged                                   # water + foam are not "floating props"
    im = Image.open(tmp_path / "smoke/reference_compare.png")
    assert im.size[1] > im.size[0] / 3 * 1.5                       # portrait tiles kept their aspect
    assert "reference_compare.png" in (tmp_path / "smoke/SMOKE.md").read_text()
