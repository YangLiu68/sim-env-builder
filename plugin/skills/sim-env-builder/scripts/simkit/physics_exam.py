"""Physics examiner: does the solver, contact model and actuation of a MuJoCo scene behave?

    python -m simkit.physics_exam scene.xml [--controller controller.py] [--task-checks task_checks.py]
                                             [--seconds 3] [--out smoke/physics_report.json]

Each check returns {"name", "group", "status": pass|warn|fail|info, "value", "expect", "details"}.

Groups
  model     compile, solver options, mass/inertia sanity, actuator gains vs timestep
  initial   penetration at the reset state
  rollout   NaN/explosion, solver warnings, max penetration, fall-through, settling, joint limits
  probes    isolated tests with the scene's own solver settings + ground material:
            free fall vs analytic, resting contact, Coulomb friction deceleration, bounce energy,
            box stacking, buoyancy (only if the scene has a fluid density)
  numerics  determinism, timestep sensitivity, real-time factor
  actuators step response of every actuator (moves its joint/body in the commanded direction?)
  task      your own checks from --task-checks (functions named check_*)

The controller module must define ``controller(model, data)`` (sets ``data.ctrl``) or
``make_controller(model) -> callable``. It is used for the rollout instead of the default "hold"
policy (position actuators hold their current position, others output 0).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

INTEGRATORS = {mujoco.mjtIntegrator.mjINT_EULER: "Euler", mujoco.mjtIntegrator.mjINT_RK4: "RK4",
               mujoco.mjtIntegrator.mjINT_IMPLICIT: "implicit", mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "implicitfast"}
CONES = {mujoco.mjtCone.mjCONE_PYRAMIDAL: "pyramidal", mujoco.mjtCone.mjCONE_ELLIPTIC: "elliptic"}
SOLVERS = {mujoco.mjtSolver.mjSOL_PGS: "PGS", mujoco.mjtSolver.mjSOL_CG: "CG", mujoco.mjtSolver.mjSOL_NEWTON: "Newton"}


def chk(name, group, status, value=None, expect=None, details=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return {"name": name, "group": group, "status": status, "value": value, "expect": expect, "details": details}


def _name(model, obj, i):
    return mujoco.mj_id2name(model, obj, i) or f"#{i}"


def _geom_desc(model, g):
    b = model.geom_bodyid[g]
    return f"{_name(model, mujoco.mjtObj.mjOBJ_GEOM, g)}@{_name(model, mujoco.mjtObj.mjOBJ_BODY, b)}"


def reset(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def load_module(path):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(mod)
    return mod


def get_controller(model, mod):
    if mod is None:
        return hold_controller(model)
    if hasattr(mod, "make_controller"):
        return mod.make_controller(model)
    if hasattr(mod, "controller"):
        return mod.controller
    raise AttributeError("controller module needs controller(model, data) or make_controller(model)")


def _is_position_actuator(model, a):
    return (model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and model.actuator_gainprm[a, 0] > 0
            and abs(model.actuator_biasprm[a, 1] + model.actuator_gainprm[a, 0]) < 1e-9)


def _act_joint(model, a):
    if int(model.actuator_trntype[a]) in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),):
        return int(model.actuator_trnid[a, 0])
    return -1


def hold_controller(model):
    """Position actuators keep the pose they had at the first call; everything else gets 0."""
    state = {}

    def ctrl(m, d):
        if "target" not in state:
            tgt = np.zeros(m.nu)
            for a in range(m.nu):
                j = _act_joint(m, a)
                if _is_position_actuator(m, a) and j >= 0 and m.jnt_type[j] in (2, 3):
                    tgt[a] = d.qpos[m.jnt_qposadr[j]]
            if m.nu:
                lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
                lim = m.actuator_ctrllimited.astype(bool)
                tgt[lim] = np.clip(tgt[lim], lo[lim], hi[lim])
            state["target"] = tgt
        d.ctrl[:] = state["target"]
    return ctrl


def ground_geom(model):
    """The main ground: the largest plane, else the largest hfield, else -1."""
    best, area = -1, -1.0
    for g in range(model.ngeom):
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            a = (model.geom_size[g, 0] or 1e6) * (model.geom_size[g, 1] or 1e6) * 10
        elif t == mujoco.mjtGeom.mjGEOM_HFIELD:
            h = model.geom_dataid[g]
            a = model.hfield_size[h, 0] * model.hfield_size[h, 1]
        else:
            continue
        if a > area:
            best, area = g, a
    return best


def free_bodies(model):
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            out.append((int(model.jnt_bodyid[j]), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])))
    return out


def actuated_body_set(model):
    s = set()
    for a in range(model.nu):
        j = _act_joint(model, a)
        b = model.jnt_bodyid[j] if j >= 0 else (model.site_bodyid[model.actuator_trnid[a, 0]] if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_SITE else -1)
        while b > 0:
            s.add(int(b))
            b = model.body_parentid[b]
    return s


# =========================================================================== model checks

def check_model(model):
    out = [chk("compile", "model", "pass", {"nbody": model.nbody, "ngeom": model.ngeom, "njnt": model.njnt, "nu": model.nu,
                                            "nmesh": model.nmesh, "nkey": model.nkey}, details="model compiled")]
    o = model.opt
    integ = INTEGRATORS.get(o.integrator, str(o.integrator))
    status, notes = "pass", []
    if o.timestep > 0.01:
        status, notes = "warn", notes + ["timestep > 10 ms: contacts and stiff actuators may be unstable"]
    if o.timestep < 2e-4:
        status, notes = "warn", notes + ["timestep < 0.2 ms: very slow"]
    if integ == "Euler" and model.nu:
        notes.append("Euler with stiff actuators/damping: 'implicitfast' is usually more stable")
    out.append(chk("solver_options", "model", status,
                   {"timestep": o.timestep, "integrator": integ, "solver": SOLVERS.get(o.solver, str(o.solver)),
                    "iterations": o.iterations, "cone": CONES.get(o.cone, str(o.cone)), "gravity": o.gravity.tolist(),
                    "density": o.density, "viscosity": o.viscosity, "wind": o.wind.tolist()},
                   details="; ".join(notes)))
    # mass / inertia
    bad, tiny, ratio = [], [], []
    flex_bodies = set(int(b) for b in model.flex_vertbodyid) if model.nflex else set()
    for b in range(1, model.nbody):
        has_dof = model.body_dofnum[b] > 0
        if not has_dof or b in flex_bodies:  # cloth/rope particles are light by design
            continue
        m = model.body_mass[b]
        if m <= 0:
            bad.append(_name(model, mujoco.mjtObj.mjOBJ_BODY, b))
        elif m < 1e-3:
            tiny.append(f"{_name(model, mujoco.mjtObj.mjOBJ_BODY, b)}={m:.2g}kg")
        p = model.body_parentid[b]
        if p > 0 and model.body_mass[p] > 0 and m > 0:
            r = max(m / model.body_mass[p], model.body_mass[p] / m)
            if r > 200:
                ratio.append(f"{_name(model, mujoco.mjtObj.mjOBJ_BODY, p)}/{_name(model, mujoco.mjtObj.mjOBJ_BODY, b)}={r:.0f}x")
    st = "fail" if bad else ("warn" if tiny or ratio else "pass")
    out.append(chk("mass_inertia", "model", st, {"massless_dynamic": bad, "tiny_mass": tiny[:10], "extreme_ratio": ratio[:10]},
                   details="massless moving bodies break the solver; mass ratios > 200x across a joint are ill-conditioned"))
    # actuator gains vs timestep
    if model.nu:
        d = mujoco.MjData(model)
        reset(model, d)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, d, M)
        risky, unlimited, zero_gear = [], [], []
        for a in range(model.nu):
            nm = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if not model.actuator_ctrllimited[a]:
                unlimited.append(nm)
            if np.allclose(model.actuator_gear[a], 0):
                zero_gear.append(nm)
            j = _act_joint(model, a)
            if _is_position_actuator(model, a) and j >= 0:
                dof = model.jnt_dofadr[j]
                inertia = M[dof, dof] + model.dof_armature[dof]
                kp = model.actuator_gainprm[a, 0] * model.actuator_gear[a, 0] ** 2
                w = math.sqrt(kp / max(inertia, 1e-12))
                if w * model.opt.timestep > 0.6:
                    risky.append(f"{nm}: omega*dt={w * model.opt.timestep:.2f} (kp={kp:.3g}, I={inertia:.2g})")
        st = "fail" if zero_gear else ("warn" if risky else "pass")
        out.append(chk("actuator_config", "model", st, {"stiff_vs_dt": risky[:10], "unlimited_ctrl": unlimited[:10], "zero_gear": zero_gear},
                       details="position gains with omega*dt > 0.6 oscillate/explode: lower kp, add armature/damping, or shrink timestep"))
    return out


# =========================================================================== initial state

def check_initial(model):
    d = mujoco.MjData(model)
    reset(model, d)
    worst = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < -0.002:
            worst.append((float(c.dist), _geom_desc(model, c.geom1), _geom_desc(model, c.geom2)))
    worst.sort()
    mx = -worst[0][0] if worst else 0.0
    st = "fail" if mx > 0.02 else ("warn" if mx > 0.005 else "pass")
    return [chk("initial_penetration", "initial", st, round(mx, 4), "< 0.005 m",
                details="; ".join(f"{a} x {b}: {-dd * 100:.1f} cm" for dd, a, b in worst[:8]) or "no overlapping geoms at reset")]


# =========================================================================== rollout

def rollout(model, controller, seconds, record_contacts=True):
    d = mujoco.MjData(model)
    reset(model, d)
    n = int(round(seconds / model.opt.timestep))
    fb = free_bodies(model)
    start = {b: d.xpos[b].copy() for b, _, _ in fb}
    max_pen, pen_pair, max_qvel = 0.0, "", 0.0
    lim_viol = {}
    blew = None
    t0 = time.perf_counter()
    for k in range(n):
        if controller is not None:
            try:
                controller(model, d)
            except Exception as e:  # noqa: BLE001
                return {"error": f"controller raised at t={d.time:.3f}: {e!r}", "data": d}
        mujoco.mj_step(model, d)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            blew = d.time
            break
        if record_contacts and k % 5 == 0:
            for i in range(d.ncon):
                c = d.contact[i]
                if -c.dist > max_pen:
                    max_pen, pen_pair = -c.dist, f"{_geom_desc(model, c.geom1)} x {_geom_desc(model, c.geom2)}"
            max_qvel = max(max_qvel, float(np.abs(d.qvel).max()) if model.nv else 0.0)
            for j in range(model.njnt):
                if model.jnt_limited[j] and model.jnt_type[j] in (2, 3):
                    q = d.qpos[model.jnt_qposadr[j]]
                    lo, hi = model.jnt_range[j]
                    v = max(lo - q, q - hi, 0)
                    if v > lim_viol.get(j, 0):
                        lim_viol[j] = v
    wall = time.perf_counter() - t0
    return {"data": d, "steps": k + 1, "wall": wall, "blew_up_at": blew, "max_pen": max_pen, "pen_pair": pen_pair,
            "max_qvel": max_qvel, "start": start, "free": fb, "limit_violation": lim_viol}


def check_rollout(model, controller, seconds, used_custom, world_factory=None):
    out = []
    r = rollout(model, controller, seconds)
    if "error" in r:
        return [chk("controller", "rollout", "fail", details=r["error"])], r
    d = r["data"]
    if r["blew_up_at"] is not None:
        out.append(chk("numerical_stability", "rollout", "fail", f"NaN at t={r['blew_up_at']:.3f}s", "finite state",
                       details="simulation diverged: check actuator gains, masses, timestep, initial penetrations"))
    else:
        st = "fail" if r["max_qvel"] > 1000 else ("warn" if r["max_qvel"] > 100 else "pass")
        out.append(chk("numerical_stability", "rollout", st, {"max_abs_qvel": round(r["max_qvel"], 2), "sim_seconds": round(d.time, 2)},
                       "max |qvel| < 100", details="state stayed finite"))
    warns = {}
    for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_CONTACTFULL, mujoco.mjtWarning.mjWARN_CNSTRFULL,
              mujoco.mjtWarning.mjWARN_INERTIA, mujoco.mjtWarning.mjWARN_BADCTRL, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL):
        cnt = d.warning[w].number
        if cnt:
            warns[str(w).split(".")[-1]] = int(cnt)
    out.append(chk("solver_warnings", "rollout", "fail" if warns else "pass", warns or 0, "none",
                   details="MuJoCo raised warnings during the rollout" if warns else "no MuJoCo warnings"))
    mp = r["max_pen"]
    st = "fail" if mp > 0.05 else ("warn" if mp > 0.01 else "pass")
    out.append(chk("max_penetration", "rollout", st, round(mp, 4), "< 0.01 m", details=r["pen_pair"] or "no contacts"))
    # fall-through / fly-away / settling for free bodies
    gg = ground_geom(model)
    gz = float(d.geom_xpos[gg][2]) if gg >= 0 else None
    act = actuated_body_set(model)
    fell, flew, moved, unsettled = [], [], [], []
    from .render import scene_bounds

    lo, hi = scene_bounds(model, d)
    span = float(np.linalg.norm(hi - lo)) + 1.0
    for b, qadr, dadr in r["free"]:
        nm = _name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        p = d.xpos[b]
        if gz is not None and model.geom_type[gg] == mujoco.mjtGeom.mjGEOM_PLANE and p[2] < gz - 0.5:
            fell.append(f"{nm} z={p[2]:.2f}")
        if np.linalg.norm(p - r["start"][b]) > 3 * span:
            flew.append(nm)
    # settling is judged with the robots HOLDING their home pose: a demo that carries or pushes
    # objects must not count as "objects did not settle"
    if used_custom:
        from .world import with_world

        rp = rollout(model, with_world(world_factory() if world_factory else None, hold_controller(model)), min(seconds, 3.0),
                     record_contacts=False)
    else:
        rp = r
    dp = rp.get("data", d)
    for b, qadr, dadr in rp.get("free", []):
        if b in act or rp.get("blew_up_at") is not None:
            continue
        nm = _name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        disp = float(np.linalg.norm(dp.xpos[b] - rp["start"][b]))
        if disp > 0.2:
            moved.append(f"{nm} {disp:.2f} m")
        sp = float(np.linalg.norm(dp.qvel[dadr:dadr + 3]))
        if sp > 0.05:
            unsettled.append(f"{nm} {sp:.2f} m/s")
    out.append(chk("fall_through", "rollout", "fail" if (fell or flew) else "pass", {"fell_below_ground": fell[:10], "flew_away": flew[:10]},
                   details="free bodies must stay on/above the ground and inside the scene"))
    if any(b not in act for b, _, _ in r["free"]):
        st = "warn" if (moved or unsettled) else "pass"
        out.append(chk("passive_settling", "rollout", st, {"moved_over_0.2m": moved[:10], "still_moving": unsettled[:10]},
                       details="with robots holding their home pose, unactuated free objects should settle where they were placed "
                               "(moving = spawned in the air, overlapping, or on a slope)"))
    viol = {(_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)): round(float(v), 4) for j, v in r["limit_violation"].items() if v > 0.02}
    out.append(chk("joint_limits", "rollout", "warn" if viol else "pass", viol or 0, "< 0.02 overshoot",
                   details="joints pushed past their range (soft limits) during the rollout"))
    return out, r


# =========================================================================== probes

def _probe_xml(model, body_xml, extra_geoms="", opt_override=None):
    o = model.opt
    gg = ground_geom(model)
    fr = model.geom_friction[gg] if gg >= 0 else np.array([1, 0.005, 0.0001])
    solref = model.geom_solref[gg] if gg >= 0 else np.array([0.02, 1])
    solimp = model.geom_solimp[gg] if gg >= 0 else np.array([0.9, 0.95, 0.001, 0.5, 2])
    condim = int(model.geom_condim[gg]) if gg >= 0 else 3
    g = o.gravity if opt_override is None or "gravity" not in opt_override else opt_override["gravity"]
    dens = o.density if opt_override is None or "density" not in opt_override else opt_override["density"]
    visc = o.viscosity if opt_override is None or "viscosity" not in opt_override else opt_override["viscosity"]
    return f"""<mujoco><option timestep="{o.timestep}" gravity="{g[0]} {g[1]} {g[2]}" integrator="{INTEGRATORS[o.integrator]}"
      cone="{CONES[o.cone]}" solver="{SOLVERS[o.solver]}" iterations="{o.iterations}" impratio="{o.impratio}"
      density="{dens}" viscosity="{visc}"/>
      <worldbody><geom name="floor" type="plane" size="50 50 .1" friction="{fr[0]} {fr[1]} {fr[2]}"
        solref="{solref[0]} {solref[1]}" solimp="{' '.join(str(x) for x in solimp)}" condim="{condim}"/>
      {extra_geoms}{body_xml}</worldbody></mujoco>""", float(fr[0])


def check_probes(model):
    out = []
    o = model.opt
    gvec = np.array(o.gravity, float)
    gmag = float(np.linalg.norm(gvec))
    dt = o.timestep
    # 1. free fall (fluid and wind off: pure gravity + integrator)
    xml, _ = _probe_xml(model, '<body name="p" pos="0 0 20"><freejoint/><geom type="sphere" size="0.1" contype="0" conaffinity="0" mass="1"/></body>',
                        opt_override={"density": 0, "viscosity": 0})
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    T = 0.5
    n = int(round(T / dt))
    for _ in range(n):
        mujoco.mj_step(m, d)
    disp = d.qpos[:3] - np.array([0, 0, 20])
    t = n * dt
    exact = 0.5 * gvec * t**2
    disc = gvec * dt**2 * n * (n + 1) / 2  # semi-implicit Euler
    err = min(np.linalg.norm(disp - exact), np.linalg.norm(disp - disc))
    tol = max(0.01 * np.linalg.norm(exact), 1e-6) + 1e-9
    if gmag < 1e-9:
        st = "pass" if np.linalg.norm(disp) < 1e-6 else "fail"
        out.append(chk("freefall_gravity", "probes", st, {"displacement": disp.round(6).tolist()}, "stays still (zero gravity)",
                       details="zero-gravity scene: a free probe must not drift"))
    else:
        st = "pass" if err <= tol else "fail"
        out.append(chk("freefall_gravity", "probes", st, {"t": t, "measured": disp.round(4).tolist(), "analytic": exact.round(4).tolist(), "err_m": round(float(err), 6)},
                       f"|err| < {tol:.4f} m", details=f"|g| = {gmag:.3f} m/s^2 along {np.round(gvec / gmag, 3).tolist()}"))
    if gmag < 1e-9:
        return out  # contact probes need gravity
    up = -gvec / gmag
    # 2. resting contact
    xml, mu = _probe_xml(model, '<body name="b" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="2" friction="0.1 0.005 0.0001"/></body>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    zs = []
    for k in range(int(2.0 / dt)):
        mujoco.mj_step(m, d)
        if k * dt > 1.0:
            zs.append(d.qpos[2])
    pen = 0.1 - float(np.min(zs))
    jitter = float(np.ptp(zs))
    drift = float(np.linalg.norm(d.qpos[:2]))
    st = "pass" if (pen < 0.005 and jitter < 5e-4 and drift < 1e-3) else ("warn" if pen < 0.02 else "fail")
    out.append(chk("resting_contact", "probes", st, {"penetration_m": round(pen, 5), "jitter_m": round(jitter, 6), "xy_drift_m": round(drift, 6)},
                   "pen < 5 mm, jitter < 0.5 mm", details="2 kg box resting on the scene's ground material"))
    # 3. Coulomb friction: slide with v0 and measure deceleration (MuJoCo uses the max friction of the pair)
    if abs(up[2]) > 0.99:
        v0 = 2.0
        xml, mu = _probe_xml(model, '<body name="b" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="2" friction="0.05 0.005 0.0001"/></body>')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        for _ in range(int(0.3 / dt)):
            mujoco.mj_step(m, d)
        d.qvel[0] = v0
        ts, vs = [], []
        for _ in range(int(3.0 / dt)):
            mujoco.mj_step(m, d)
            ts.append(d.time)
            vs.append(d.qvel[0])
            if d.qvel[0] < 0.05:
                break
        ts, vs = np.array(ts), np.array(vs)
        sel = (vs > 0.2 * v0) & (vs < 0.9 * v0)
        if sel.sum() > 5 and mu > 0:
            a = -np.polyfit(ts[sel], vs[sel], 1)[0]
            expect = mu * gmag
            rel = abs(a - expect) / expect
            st = "pass" if rel < 0.15 else ("warn" if rel < 0.35 else "fail")
            out.append(chk("coulomb_friction", "probes", st, {"decel": round(float(a), 3), "mu_g": round(expect, 3), "rel_err": round(float(rel), 3)},
                           "within 15% of mu*g", details=f"ground mu={mu:.2f} ({CONES[o.cone]} cone)"))
        else:
            out.append(chk("coulomb_friction", "probes", "warn" if mu > 0 else "info", {"mu": mu}, details="could not measure (frictionless or stopped instantly)"))
    # 4. bounce energy
    xml, _ = _probe_xml(model, '<body name="s" pos="0 0 1.0"><freejoint/><geom type="sphere" size="0.05" mass="0.2"/></body>',
                        opt_override={"density": 0, "viscosity": 0})
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    zmax_after, hit = 0.0, False
    for _ in range(int(2.5 / dt)):
        mujoco.mj_step(m, d)
        if d.ncon:
            hit = True
        if hit:
            zmax_after = max(zmax_after, d.qpos[2])
    gain = (zmax_after - 0.05) / 0.95
    st = "pass" if gain <= 1.02 and np.isfinite(gain) else "fail"
    out.append(chk("bounce_energy", "probes", st, {"rebound_ratio": round(float(gain), 3), "final_z": round(float(d.qpos[2]), 4)},
                   "rebound <= 1.02 x drop height", details="contacts must not inject energy"))
    # 5. stacking
    stack = "".join(f'<body name="s{i}" pos="0 0 {0.1 + i * 0.2 + 0.001 * i}"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>' for i in range(3))
    xml, _ = _probe_xml(model, stack)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(m, d)
    top = d.qpos[14:17]
    dxy = float(np.linalg.norm(top[:2]))
    dz = abs(float(top[2]) - 0.5)
    st = "pass" if dxy < 0.01 and dz < 0.01 else ("warn" if dxy < 0.05 and dz < 0.03 else "fail")
    out.append(chk("box_stack", "probes", st, {"top_xy_drift_m": round(dxy, 5), "top_z_err_m": round(dz, 5)}, "< 1 cm",
                   details="3 stacked 20 cm boxes must stay put for 2 s"))
    # 6. buoyancy (only for fluid scenes)
    if o.density > 0:
        xml, _ = _probe_xml(model, f'<body name="f" pos="0 0 5"><freejoint/><geom type="sphere" size="0.2" density="{o.density * 0.5}" contype="0" conaffinity="0"/></body>'
                                   f'<body name="k" pos="2 0 5"><freejoint/><geom type="sphere" size="0.2" density="{o.density * 3}" contype="0" conaffinity="0"/></body>')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        for _ in range(int(1.0 / dt)):
            mujoco.mj_step(m, d)
        light_dz, heavy_dz = d.qpos[2] - 5, d.qpos[9] - 5
        st = "pass" if (light_dz * up[2] > 0 and heavy_dz * up[2] < 0) else "fail"
        out.append(chk("buoyancy", "probes", st, {"light_sphere_dz": round(float(light_dz), 3), "heavy_sphere_dz": round(float(heavy_dz), 3)},
                       "light rises, heavy sinks", details=f"fluid density {o.density} kg/m^3, viscosity {o.viscosity}"))
    return out


# =========================================================================== numerics

def check_numerics(model, controller_factory, seconds=1.0):
    out = []
    a = rollout(model, controller_factory(), seconds, record_contacts=False)
    b = rollout(model, controller_factory(), seconds, record_contacts=False)
    if "error" in a or "error" in b:
        return [chk("determinism", "numerics", "fail", details=a.get("error") or b.get("error"))]
    same = np.array_equal(a["data"].qpos, b["data"].qpos)
    out.append(chk("determinism", "numerics", "pass" if same else "fail", same, True, details="two identical runs give identical states"))
    # timestep sensitivity on free bodies
    fb = free_bodies(model)
    if fb:
        import copy

        m2 = copy.deepcopy(model)
        if m2 is not None:
            m2.opt.timestep = model.opt.timestep / 2
            c = rollout(m2, controller_factory(), seconds, record_contacts=False)
            if "error" not in c and c["blew_up_at"] is None and a["blew_up_at"] is None:
                diffs = [float(np.linalg.norm(a["data"].xpos[bb] - c["data"].xpos[bb])) for bb, _, _ in fb]
                mx = max(diffs)
                out.append(chk("timestep_sensitivity", "numerics", "pass" if mx < 0.05 else "warn", round(mx, 4), "< 0.05 m",
                               details="free-body positions after 1 s at dt vs dt/2 (contact-rich scenes can legitimately diverge)"))
            elif c.get("blew_up_at") is not None:
                out.append(chk("timestep_sensitivity", "numerics", "fail", details="diverged at dt/2"))
    rtf = (a["data"].time / a["wall"]) if a["wall"] > 0 else float("inf")
    out.append(chk("realtime_factor", "numerics", "pass" if rtf >= 1 else "warn", round(rtf, 2), ">= 1x",
                   details=f"{a['steps'] / max(a['wall'], 1e-9):.0f} steps/s on this machine"))
    return out


# =========================================================================== actuators

def check_actuators(model, settle=0.3, step_time=0.5, limit=40, world_factory=None):
    out = []
    if not model.nu:
        return out
    base = mujoco.MjData(model)
    reset(model, base)
    hold = hold_controller(model)
    world = world_factory() if world_factory else None
    for _ in range(int(settle / model.opt.timestep)):
        if world is not None:
            world(model, base)
        hold(model, base)
        mujoco.mj_step(model, base)
    target0 = base.ctrl.copy()
    dead, wrong, ok, unstable = [], [], 0, []
    for a in range(min(model.nu, limit)):
        nm = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        d = mujoco.MjData(model)
        d.qpos[:], d.qvel[:], d.act[:] = base.qpos, base.qvel, base.act
        d.time = base.time
        mujoco.mj_forward(model, d)
        lo, hi = model.actuator_ctrlrange[a] if model.actuator_ctrllimited[a] else (-1.0, 1.0)
        ctrl = target0.copy()
        j = _act_joint(model, a)
        if _is_position_actuator(model, a):
            span = hi - lo
            cur = ctrl[a]
            new = cur + 0.25 * span if cur + 0.25 * span <= hi else cur - 0.25 * span
            ctrl[a] = new
            sign = np.sign(new - cur)
        else:
            new = hi * 0.6 if abs(hi) >= abs(lo) else lo * 0.6
            if lo < 0 < hi:
                new = hi * 0.6
            ctrl[a] = new
            sign = np.sign(new - target0[a]) * np.sign(model.actuator_gear[a, 0] or 1)
        # measure
        if j >= 0:
            qa = model.jnt_qposadr[j]
            jt = model.jnt_type[j]
            q0 = d.qpos[qa:qa + (7 if jt == 0 else 4 if jt == 1 else 1)].copy()
        b = model.site_bodyid[model.actuator_trnid[a, 0]] if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_SITE else (model.jnt_bodyid[j] if j >= 0 else 0)
        p0 = d.xpos[b].copy()
        tl0 = d.ten_length[model.actuator_trnid[a, 0]] if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON else None
        blew = False
        for _ in range(int(step_time / model.opt.timestep)):
            if world is not None:
                world(model, d)
            d.ctrl[:] = ctrl
            mujoco.mj_step(model, d)
            if not np.isfinite(d.qpos).all():
                blew = True
                break
        if blew:
            unstable.append(nm)
            continue
        force = float(np.abs(d.actuator_force[a]))
        if j >= 0 and model.jnt_type[j] in (2, 3):
            dq = float(d.qpos[qa] - q0[0])
            resp, signed = abs(dq), np.sign(dq)
            is_dead = resp < 1e-4
        elif tl0 is not None:
            dl = float(d.ten_length[model.actuator_trnid[a, 0]] - tl0)
            resp, signed = abs(dl), None
            is_dead = resp < 1e-4 and force < 1e-6
        else:
            # site / body transmissions (thrusters, adhesion): one thruster alone may not lift its
            # body, so judge by the force it applies to the generalized coordinates
            resp, signed = float(np.linalg.norm(d.xpos[b] - p0)), None
            is_dead = force < 1e-6 and resp < 1e-4
        if is_dead:
            dead.append(nm)
        elif signed is not None and sign != 0 and signed != sign and model.actuator_gear[a, 0] > 0:
            wrong.append(f"{nm} (moved {signed:+.0f}, commanded {sign:+.0f})")
        else:
            ok += 1
    tested = min(model.nu, limit)
    st = "fail" if (dead or unstable) else ("warn" if wrong else "pass")
    out.append(chk("actuator_response", "actuators", st, {"responsive": ok, "tested": tested, "dead": dead, "wrong_direction": wrong, "unstable": unstable},
                   "every actuator moves its target in the commanded direction",
                   details="step command from a settled state for 0.5 s; 'dead' = < 0.1 mm/mrad response (blocked, zero gear, or too weak)"))
    return out


# =========================================================================== main

def examine(xml_path, controller_path=None, task_checks_path=None, seconds=3.0):
    report = {"scene": str(xml_path), "checks": []}
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:  # noqa: BLE001
        report["checks"].append(chk("compile", "model", "fail", details=str(e)))
        report["summary"] = summarize(report["checks"])
        return report
    ctrl_mod = load_module(controller_path) if controller_path else None
    from .world import load_world, with_world

    def world_factory():  # world.py dynamics (buoyancy, conveyors, NPCs) run in EVERY rollout
        return load_world(Path(xml_path).resolve().parent, model)

    def factory():
        return with_world(world_factory(), get_controller(model, ctrl_mod))
    sections = [
        ("model", lambda: check_model(model)),
        ("initial", lambda: check_initial(model)),
        ("rollout", lambda: check_rollout(model, factory(), seconds, ctrl_mod is not None, world_factory)[0]),
        ("probes", lambda: check_probes(model)),
        ("numerics", lambda: check_numerics(model, factory)),
        ("actuators", lambda: check_actuators(model, world_factory=world_factory)),
    ]
    for group, fn in sections:
        try:
            report["checks"].extend(fn())
        except Exception as e:  # noqa: BLE001
            report["checks"].append(chk(f"{group}_crashed", group, "fail", details=f"{e!r}\n{traceback.format_exc()[-800:]}"))
    if task_checks_path:
        mod = load_module(task_checks_path)
        for name in sorted(n for n in dir(mod) if n.startswith("check_")):
            try:
                d = mujoco.MjData(model)
                reset(model, d)
                res = getattr(mod, name)(model, d)
                res = res if isinstance(res, dict) else {"status": "pass" if res else "fail"}
                report["checks"].append(chk(name[6:], "task", res.get("status", "info"), res.get("value"), res.get("expect"), res.get("details", "")))
            except Exception as e:  # noqa: BLE001
                report["checks"].append(chk(name[6:], "task", "fail", details=f"crashed: {e!r}"))
    report["summary"] = summarize(report["checks"])
    return report


def summarize(checks):
    s = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "warn", "fail", "info")}
    s["verdict"] = "FAIL" if s["fail"] else ("PASS_WITH_WARNINGS" if s["warn"] else "PASS")
    return s


def print_report(rep):
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}
    for c in rep["checks"]:
        v = json.dumps(c["value"]) if c["value"] is not None else ""
        print(f"[{icon[c['status']]}] {c['group']:9s} {c['name']:22s} {v[:110]}")
        if c["status"] in ("warn", "fail") and c["details"]:
            print(f"        -> {c['details'][:300]}")
    print("verdict:", rep["summary"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--controller")
    ap.add_argument("--task-checks")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = examine(a.scene, a.controller, a.task_checks, a.seconds)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print_report(rep)
    return 1 if rep["summary"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
