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
