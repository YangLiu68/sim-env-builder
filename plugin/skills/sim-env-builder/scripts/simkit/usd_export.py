"""Export a compiled MuJoCo scene to a USD stage with UsdPhysics for NVIDIA Isaac Sim.

    python -m simkit.usd_export scene.xml isaac/scene.usda      # export + validate

Why this route: Isaac Sim needs an NVIDIA RTX GPU (Linux/Windows). Building and smoke-testing
the scene in MuJoCo first works on any machine; this exporter then produces an equivalent stage:

* bodies -> Xform prims (flattened under /World/<root>, world poses; PhysX forbids nested rigid
  bodies), moving bodies get RigidBodyAPI + MassAPI (mass, COM, diagonal inertia, principal axes)
* geoms -> Cube/Sphere/Capsule/Cylinder/Mesh prims with displayColor; colliding geoms get
  CollisionAPI (meshes: convexHull approximation); planes -> 200 m thin box; hfields -> mesh
* joints -> Revolute / Prismatic / Spherical joints with limits (degrees for revolute) and
  DriveAPI from position actuators (stiffness=kp, damping=kv); actuator metadata is stored in
  customData ``mjc:actuators`` so a controller script can map ctrl -> drive targets/efforts
* robots fixed to the world get a FixedJoint to the world; each actuated tree gets ArticulationRootAPI
* cameras -> UsdGeom.Camera (same -Z forward / +Y up convention), lights -> UsdLux
* gravity -> UsdPhysics.Scene

``validate(usd_path, xml_path)`` reopens the stage and checks prim counts, masses, joint targets
and bounds against MuJoCo. It cannot run PhysX or RTX rendering: run ``isaac_smoke.py`` (template
in the skill) on an Isaac Sim machine for that.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name or "unnamed")
    return s if not s[0].isdigit() else "_" + s


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_from_x_to(v):
    """Quaternion (w, x, y, z) rotating +X onto unit vector v."""
    x = np.array([1.0, 0, 0])
    v = np.asarray(v, float) / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(x, v))
    if c > 0.999999:
        return np.array([1.0, 0, 0, 0])
    if c < -0.999999:
        return np.array([0.0, 0, 0, 1])
    axis = np.cross(x, v)
    axis /= np.linalg.norm(axis)
    ang = math.acos(max(-1.0, min(1.0, c)))
    return np.array([math.cos(ang / 2), *(axis * math.sin(ang / 2))])


def export(xml_path, usd_path, keyframe: int | None = 0) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nkey and keyframe is not None:
        mujoco.mj_resetDataKeyframe(model, data, keyframe)
    mujoco.mj_forward(model, data)
    usd_path = Path(usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    g = np.array(model.opt.gravity, float)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    gm = float(np.linalg.norm(g))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*(g / gm if gm > 0 else np.array([0, 0, -1.0]))))
    scene.CreateGravityMagnitudeAttr(gm)
    scene.GetPrim().SetCustomDataByKey("mjc:timestep", float(model.opt.timestep))

    def top(b):
        while b and model.body_parentid[b] != 0:
            b = model.body_parentid[b]
        return b

    names = {}
    for b in range(model.nbody):
        nm = _safe(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
        while nm in names.values():
            nm += "_"
        names[b] = nm
    moving = {b for b in range(1, model.nbody) if _moves(model, b)}
    paths = {0: "/World/static"}
    UsdGeom.Xform.Define(stage, "/World/static")
    for b in range(1, model.nbody):
        root = top(b)
        group = f"/World/{names[root]}"
        if not stage.GetPrimAtPath(group):
            UsdGeom.Xform.Define(stage, group)
        p = f"{group}/{names[b]}" if b != root else f"{group}/{names[b]}_link"
        paths[b] = p
        xf = UsdGeom.Xform.Define(stage, p)
        xf.AddTranslateOp().Set(Gf.Vec3d(*data.xpos[b]))
        w, x, y, z = data.xquat[b]
        xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
        prim = xf.GetPrim()
        if b in moving and model.body_mass[b] > 0:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(float(model.body_mass[b]))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*model.body_ipos[b]))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*model.body_inertia[b]))
            iq = model.body_iquat[b]
            mass.CreatePrincipalAxesAttr(Gf.Quatf(float(iq[0]), float(iq[1]), float(iq[2]), float(iq[3])))

    counts = {"geoms": 0, "colliders": 0, "meshes": 0, "joints": 0, "drives": 0, "cameras": 0}
    for gi in range(model.ngeom):
        b = model.geom_bodyid[gi]
        gt = model.geom_type[gi]
        parent = paths[b]
        gname = _safe(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"g{gi}")
        gp = f"{parent}/{gname}_{gi}"
        size = model.geom_size[gi]
        rgba = model.mat_rgba[model.geom_matid[gi]] if model.geom_matid[gi] >= 0 else model.geom_rgba[gi]
        collides = bool(model.geom_contype[gi] or model.geom_conaffinity[gi])
        visual_only_hidden = model.geom_group[gi] >= 3 and collides  # collision proxies: invisible
        if b == 0:
            pos, quat = data.geom_xpos[gi], np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.geom_xmat[gi])
        else:
            pos, quat = model.geom_pos[gi], model.geom_quat[gi]
        prim = None
        if gt == mujoco.mjtGeom.mjGEOM_BOX:
            c = UsdGeom.Cube.Define(stage, gp)
            c.CreateSizeAttr(2.0)
            prim, scale = c, size[:3]
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            c = UsdGeom.Sphere.Define(stage, gp)
            c.CreateRadiusAttr(float(size[0]))
            prim, scale = c, None
        elif gt == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            c = UsdGeom.Sphere.Define(stage, gp)
            c.CreateRadiusAttr(1.0)
            prim, scale = c, size[:3]
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c = UsdGeom.Capsule.Define(stage, gp)
            c.CreateRadiusAttr(float(size[0]))
            c.CreateHeightAttr(float(2 * size[1]))
            c.CreateAxisAttr("Z")
            prim, scale = c, None
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            c = UsdGeom.Cylinder.Define(stage, gp)
            c.CreateRadiusAttr(float(size[0]))
            c.CreateHeightAttr(float(2 * size[1]))
            c.CreateAxisAttr("Z")
            prim, scale = c, None
        elif gt == mujoco.mjtGeom.mjGEOM_PLANE:
            sx = float(size[0]) or 100.0
            sy = float(size[1]) or 100.0
            c = UsdGeom.Cube.Define(stage, gp)
            c.CreateSizeAttr(2.0)
            prim, scale = c, (sx, sy, 0.05)
            pos = np.asarray(pos) - np.array([0, 0, 0.05]) if b == 0 else pos
        elif gt == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gi]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            verts = model.mesh_vert[va:va + vn]
            faces = model.mesh_face[fa:fa + fn]
            c = UsdGeom.Mesh.Define(stage, gp)
            c.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
            c.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
            c.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.ravel()]))
            c.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            prim, scale = c, None
            counts["meshes"] += 1
        elif gt == mujoco.mjtGeom.mjGEOM_HFIELD:
            hid = model.geom_dataid[gi]
            nr, nc = model.hfield_nrow[hid], model.hfield_ncol[hid]
            sx, sy, sz, _ = model.hfield_size[hid]
            hd = model.hfield_data[model.hfield_adr[hid]: model.hfield_adr[hid] + nr * nc].reshape(nr, nc)
            step = max(1, max(nr, nc) // 160)
            rows, cols = range(0, nr, step), range(0, nc, step)
            pts = [Gf.Vec3f(float(-sx + 2 * sx * j / (nc - 1)), float(-sy + 2 * sy * i / (nr - 1)), float(hd[i, j] * sz)) for i in rows for j in cols]
            R, C = len(rows), len(cols)
            idx = []
            for i in range(R - 1):
                for j in range(C - 1):
                    a = i * C + j
                    idx += [a, a + 1, a + C + 1, a + C]
            c = UsdGeom.Mesh.Define(stage, gp)
            c.CreatePointsAttr(Vt.Vec3fArray(pts))
            c.CreateFaceVertexCountsAttr(Vt.IntArray([4] * ((R - 1) * (C - 1))))
            c.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
            prim, scale = c, None
        else:
            continue
        counts["geoms"] += 1
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        xf.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
        if scale is not None:
            xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
        prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, rgba[:3]))]))
        if rgba[3] < 1:
            prim.CreateDisplayOpacityAttr(Vt.FloatArray([float(rgba[3])]))
        if visual_only_hidden:
            prim.CreatePurposeAttr(UsdGeom.Tokens.guide)
        if collides:
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            counts["colliders"] += 1
            if int(gt) in (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_HFIELD),):
                mc = UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
                mc.CreateApproximationAttr("convexHull" if gt == mujoco.mjtGeom.mjGEOM_MESH else "none")
            fr = model.geom_friction[gi]
            prim.GetPrim().SetCustomDataByKey("mjc:friction", Vt.FloatArray([float(x) for x in fr]))

    # joints + drives
    acts_by_joint: dict[int, list] = {}
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),):
            acts_by_joint.setdefault(int(model.actuator_trnid[a, 0]), []).append(a)
    roots_with_joints = set()
    for j in range(model.njnt):
        jt = model.jnt_type[j]
        b = model.jnt_bodyid[j]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            continue
        roots_with_joints.add(top(b))
        parent = model.body_parentid[b]
        jname = _safe(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}")
        jp = f"{paths[b]}/{jname}"
        if jt == mujoco.mjtJoint.mjJNT_HINGE:
            J = UsdPhysics.RevoluteJoint.Define(stage, jp)
        elif jt == mujoco.mjtJoint.mjJNT_SLIDE:
            J = UsdPhysics.PrismaticJoint.Define(stage, jp)
        else:
            J = UsdPhysics.SphericalJoint.Define(stage, jp)
        if parent != 0:
            J.CreateBody0Rel().SetTargets([Sdf.Path(paths[parent])])
        J.CreateBody1Rel().SetTargets([Sdf.Path(paths[b])])
        # anchor: joint position in child frame -> world -> parent frame
        anchor_w = data.xanchor[j]
        axis_w = data.xaxis[j]
        child_R = data.xmat[b].reshape(3, 3)
        lp1 = child_R.T @ (anchor_w - data.xpos[b])
        if parent != 0:
            par_R = data.xmat[parent].reshape(3, 3)
            lp0 = par_R.T @ (anchor_w - data.xpos[parent])
            ax0 = par_R.T @ axis_w
        else:
            lp0, ax0 = anchor_w, axis_w
        ax1 = child_R.T @ axis_w
        q0, q1 = _quat_from_x_to(ax0), _quat_from_x_to(ax1)
        J.CreateLocalPos0Attr(Gf.Vec3f(*map(float, lp0)))
        J.CreateLocalPos1Attr(Gf.Vec3f(*map(float, lp1)))
        J.CreateLocalRot0Attr(Gf.Quatf(*map(float, q0)))
        J.CreateLocalRot1Attr(Gf.Quatf(*map(float, q1)))
        if int(jt) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE),):
            J.CreateAxisAttr("X")
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                if jt == mujoco.mjtJoint.mjJNT_HINGE:
                    lo, hi = math.degrees(lo), math.degrees(hi)
                J.CreateLowerLimitAttr(float(lo))
                J.CreateUpperLimitAttr(float(hi))
            kind = "angular" if jt == mujoco.mjtJoint.mjJNT_HINGE else "linear"
            meta = []
            for a in acts_by_joint.get(j, []):
                gain, bias = model.actuator_gainprm[a], model.actuator_biasprm[a]
                is_pos = model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and abs(bias[1] + gain[0]) < 1e-9
                drive = UsdPhysics.DriveAPI.Apply(J.GetPrim(), kind)
                gear = float(model.actuator_gear[a, 0])
                if is_pos:
                    kp, kv = float(gain[0]), float(-bias[2])
                    if kind == "angular":  # PhysX angular drives use degrees
                        kp, kv = kp * math.pi / 180, kv * math.pi / 180
                    drive.CreateStiffnessAttr(kp)
                    drive.CreateDampingAttr(kv)
                    drive.CreateTypeAttr("force")
                    tgt = float(data.ctrl[a]) if model.nu else 0.0
                    drive.CreateTargetPositionAttr(math.degrees(tgt) if kind == "angular" else tgt)
                else:
                    drive.CreateStiffnessAttr(0.0)
                    drive.CreateDampingAttr(0.0)
                if model.actuator_forcelimited[a]:
                    drive.CreateMaxForceAttr(float(abs(model.actuator_forcerange[a]).max()))
                counts["drives"] += 1
                meta.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}",
                             "type": "position" if is_pos else "effort", "gear": gear,
                             "ctrlrange": model.actuator_ctrlrange[a].tolist() if model.actuator_ctrllimited[a] else None})
            if meta:
                J.GetPrim().SetCustomDataByKey("mjc:actuators", json.dumps(meta))
        counts["joints"] += 1
    # articulation roots and world anchoring for fixed-base robots
    for root in roots_with_joints:
        group = stage.GetPrimAtPath(f"/World/{names[root]}")
        UsdPhysics.ArticulationRootAPI.Apply(group)
        has_free = any(model.jnt_bodyid[j] == root and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(model.njnt))
        root_has_dof = model.body_dofnum[root] > 0
        if not has_free and not root_has_dof:
            fj = UsdPhysics.FixedJoint.Define(stage, f"/World/{names[root]}/world_anchor")
            fj.CreateBody1Rel().SetTargets([Sdf.Path(paths[root])])
            fj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, data.xpos[root])))
            w, x, y, z = data.xquat[root]
            fj.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    # sites with thrust actuators -> metadata only (apply forces in the controller script)
    site_acts = [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a), "site": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.actuator_trnid[a, 0]),
                  "body_prim": paths[model.site_bodyid[model.actuator_trnid[a, 0]]], "gear": model.actuator_gear[a].tolist(),
                  "site_pos": model.site_pos[model.actuator_trnid[a, 0]].tolist()}
                 for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_SITE]
    if site_acts:
        world.GetPrim().SetCustomDataByKey("mjc:site_actuators", json.dumps(site_acts))
    # cameras
    for c in range(model.ncam):
        cname = _safe(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c) or f"cam{c}")
        cam = UsdGeom.Camera.Define(stage, f"/World/cameras/{cname}")
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.cam_xmat[c])
        xf = UsdGeom.Xformable(cam)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, data.cam_xpos[c])))
        xf.AddOrientOp().Set(Gf.Quatf(*map(float, q)))
        fovy = float(model.cam_fovy[c])
        vert = 2 * 20.0 * math.tan(math.radians(fovy) / 2)
        cam.CreateFocalLengthAttr(20.0)
        cam.CreateVerticalApertureAttr(vert)
        cam.CreateHorizontalApertureAttr(vert * 16 / 9)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 5000))
        counts["cameras"] += 1
    sun = UsdLux.DistantLight.Define(stage, "/World/lights/sun")
    sun.CreateIntensityAttr(3000)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 20, 0))
    dome = UsdLux.DomeLight.Define(stage, "/World/lights/sky")
    dome.CreateIntensityAttr(600)
    stage.GetRootLayer().Save()
    return {"usd": str(usd_path), **counts, "bodies": model.nbody - 1, "moving_bodies": len(moving)}


def _moves(model, b):
    while b:
        if model.body_dofnum[b] > 0:
            return True
        b = model.body_parentid[b]
    return False


def validate(usd_path, xml_path) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    problems = []
    rigid, joints, colliders = 0, 0, 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid += 1
            m = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if not m or m <= 0:
                problems.append(f"{prim.GetPath()}: rigid body without positive mass")
            anc = prim.GetParent()
            while anc and anc.GetPath() != "/":
                if anc.HasAPI(UsdPhysics.RigidBodyAPI):
                    problems.append(f"{prim.GetPath()}: nested rigid body under {anc.GetPath()}")
                anc = anc.GetParent()
        if prim.IsA(UsdPhysics.Joint):
            joints += 1
            j = UsdPhysics.Joint(prim)
            for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
                for t in rel.GetTargets():
                    if not stage.GetPrimAtPath(t):
                        problems.append(f"{prim.GetPath()}: target {t} missing")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders += 1
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    expect_rigid = sum(1 for b in range(1, model.nbody) if _moves(model, b) and model.body_mass[b] > 0)
    expect_joints = sum(1 for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE)
    if rigid != expect_rigid:
        problems.append(f"rigid bodies {rigid} != moving MuJoCo bodies {expect_rigid}")
    if joints < expect_joints:
        problems.append(f"joints {joints} < MuJoCo joints {expect_joints}")
    return {"ok": not problems, "rigid_bodies": rigid, "joints": joints, "colliders": colliders,
            "bounds_min": list(rng.GetMin()), "bounds_max": list(rng.GetMax()), "problems": problems[:20]}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    info = export(src, dst)
    print(json.dumps(info, indent=1))
    print(json.dumps(validate(dst, src), indent=1))
