"""Tested procedural robots (MJCF) + matching controllers. Customise freely; keep the tuning rules.

Each builder returns a ``Robot`` with ``body`` (worldbody XML), ``actuators``, ``sensors``,
``contacts`` (excludes), ``home`` ({joint: value}, turned into the "home" keyframe by Scene.write) and ``info``.

    from simkit import robots as R
    quad = R.quadrotor("drone", pos=(0, 0, 0.2))
    quad.add_to(scene)                                   # body + actuators + sensors + excludes
    ctrl = R.QuadrotorController(model, "drone"); ctrl.target = (2, 0, 1.5)   # ctrl(model, data)

Builders: quadrotor, car (Ackermann, 4 wheels), diff_drive (mobile base), arm (n-DoF + parallel
gripper, gravity-compensated), quadruped (12-DoF). Controllers: QuadrotorController,
CarController (speed + pure pursuit), DiffDriveController (go-to-goal), ArmController (DLS IK +
gripper), StandController (hold keyframe pose, any position-actuated robot).

Tuning rules baked in (see reference/robots.md): implicitfast integrator, armature on every
actuated joint, position-gain omega*dt < 0.6, gravcomp on arm links, explicit inertials for
visual-mesh bodies, contact excludes between adjacent links welded to the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import lookat as _lookat


@dataclass
class Robot:
    name: str
    body: str
    actuators: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)
    home: dict = field(default_factory=dict)  # {joint_name: value} for the "home" keyframe
    info: dict = field(default_factory=dict)

    def add_to(self, scene):
        """Add body, actuators, sensors, excludes; register the home pose (Scene.write builds the
        "home" keyframe by joint NAME, so body order in the XML never matters)."""
        scene.home.update(self.home)
        scene.body(self.body)
        for a in self.actuators:
            scene.actuator(a)
        for s in self.sensors:
            scene.sensor(s)
        for c in self.contacts:
            scene.contact(c)
        return self


def _v(x):
    return " ".join(f"{float(a):g}" for a in x)


# =========================================================================== quadrotor

def quadrotor(name="drone", pos=(0, 0, 0.15), mass=1.0, arm=0.17, visual_xml: str = "", color=(0.15, 0.15, 0.18, 1),
              thrust_margin=2.5, yaw_coeff=0.016) -> Robot:
    """X-configuration quadrotor. Rotors r0..r3 at (+,+), (-,+), (-,-), (+,-); r0/r2 spin CCW.

    Actuator i = thrust (N) of rotor i along body +z, plus reaction yaw torque +-yaw_coeff*thrust.
    ``visual_xml``: optional geoms (e.g. a Poly Pizza drone via ``asset.geoms_xml(collision=False)``)
    replacing the default frame look; the collision box and inertial stay.
    """
    g = 9.81
    tmax = thrust_margin * mass * g / 4
    a = arm / math.sqrt(2)
    rotors = [(a, a), (-a, a), (-a, -a), (a, -a)]
    spin = [1, -1, 1, -1]
    c = _v(color)
    look = visual_xml or "".join(
        [f'<geom type="box" size="0.07 0.05 0.025" rgba="{c}" contype="0" conaffinity="0" group="2" mass="0"/>']
        + [f'<geom type="capsule" fromto="0 0 0 {x:.4f} {y:.4f} 0" size="0.012" rgba="{c}" contype="0" conaffinity="0" group="2" mass="0"/>' for x, y in rotors]
        + [f'<geom type="cylinder" size="0.065 0.004" pos="{x:.4f} {y:.4f} 0.03" rgba="0.7 0.75 0.8 0.5" contype="0" conaffinity="0" group="2" mass="0"/>' for x, y in rotors]
        + [f'<geom type="cylinder" size="0.012 0.04" pos="{x:.4f} {y:.4f} -0.03" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2" mass="0"/>' for x, y in rotors])
    I = (mass * 0.0105, mass * 0.0105, mass * 0.018)
    body = (f'<body name="{name}" pos="{_v(pos)}"><freejoint name="{name}_root"/>'
            f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{_v(I)}"/>{look}'
            f'<geom name="{name}_hull" type="box" size="{arm * 0.8:.3f} {arm * 0.8:.3f} 0.04" rgba="1 0 0 0.25" group="3" mass="0"/>'
            + "".join(f'<site group="4" name="{name}_r{i}" pos="{x:.4f} {y:.4f} 0.03" size="0.01"/>' for i, (x, y) in enumerate(rotors))
            + f'<site group="4" name="{name}_imu" pos="0 0 0"/><camera name="{name}_fpv" pos="{arm * 0.95:.3f} 0 0.03" xyaxes="{_lookat((arm * 0.95, 0, 0.03), (arm * 0.95 + 1, 0, -0.12), (0, 0, 1))}" fovy="85"/></body>')
    acts = [f'<motor name="{name}_thrust{i}" site="{name}_r{i}" gear="0 0 1 0 0 {spin[i] * yaw_coeff}" ctrlrange="0 {tmax:.3f}"/>' for i in range(4)]
    sens = [f'<gyro name="{name}_gyro" site="{name}_imu"/>', f'<accelerometer name="{name}_acc" site="{name}_imu"/>',
            f'<framequat name="{name}_quat" objtype="site" objname="{name}_imu"/>']
    return Robot(name, body, acts, sens, home={},
                 info={"mass": mass, "hover_thrust_each": mass * g / 4, "max_thrust_each": tmax, "rotors": rotors, "spin": spin,
                       "yaw_coeff": yaw_coeff, "fpv_camera": f"{name}_fpv"})


class QuadrotorController:
    """Geometric position controller (Lee et al.) + rotor mixer. Set ``.target`` (x, y, z) and ``.yaw``."""

    def __init__(self, model, name="drone", kp=(4.0, 4.0, 6.0), kd=(3.0, 3.0, 4.0), kR=200.0, kw=25.0):
        self.m = model
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_root")
        self.qa, self.da = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        self.acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_thrust{i}") for i in range(4)]
        self.mass = float(model.body_subtreemass[self.bid])
        self.I = np.array(model.body_inertia[self.bid])
        sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_r{i}") for i in range(4)]
        P = np.array([model.site_pos[s][:2] for s in sites])
        yaw = np.array([model.actuator_gear[a][5] for a in self.acts])
        # [T, tx, ty, tz] = A @ f
        self.A = np.vstack([np.ones(4), P[:, 1], -P[:, 0], yaw])
        self.Ainv = np.linalg.inv(self.A)
        self.lo = model.actuator_ctrlrange[self.acts, 0]
        self.hi = model.actuator_ctrlrange[self.acts, 1]
        self.kp, self.kd, self.kR, self.kw = np.array(kp), np.array(kd), kR, kw
        self.target = np.array(model.body_pos[self.bid], float) + np.array([0, 0, 1.0])
        self.target_vel = np.zeros(3)
        self.yaw = 0.0
        self.g = -np.array(model.opt.gravity)

    def __call__(self, model, data):
        p = data.qpos[self.qa:self.qa + 3]
        v = data.qvel[self.da:self.da + 3]
        R = data.xmat[self.bid].reshape(3, 3)
        w = data.qvel[self.da + 3:self.da + 6]  # body-frame angular velocity for free joints
        a_des = self.kp * (np.asarray(self.target) - p) + self.kd * (self.target_vel - v) + self.g
        n = np.linalg.norm(a_des)
        zb = a_des / n if n > 1e-6 else np.array([0, 0, 1.0])
        T = self.mass * float(a_des @ R[:, 2])
        xc = np.array([math.cos(self.yaw), math.sin(self.yaw), 0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb) + 1e-9
        Rd = np.column_stack([np.cross(yb, zb), yb, zb])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        tau = self.I * (-self.kR * eR - self.kw * w) + np.cross(w, self.I * w)
        f = self.Ainv @ np.array([T, *tau])
        data.ctrl[self.acts] = np.clip(f, self.lo, self.hi)


# =========================================================================== car

def car(name="car", pos=(0, 0, 0), yaw_deg=0.0, length=4.5, width=1.8, mass=1300.0, wheel_radius=0.33,
        visual_xml: str = "", color=(0.15, 0.35, 0.75, 1), max_steer_deg=35.0, max_torque=600.0, max_brake=1500.0) -> Robot:
    """Rear-wheel-drive car with Ackermann-ish steering (both front knuckles share one target).

    Actuators: ``{name}_steer_l/_r`` (position, rad), ``{name}_drive_l/_r`` (motor, N*m at the
    rear wheels), ``{name}_brake_fl/fr/rl/rr`` (motor, N*m; set them with ``car_brake_torque`` so the
    torque opposes wheel spin and fades to 0 at standstill - braking never drives the car backwards).
    Use CarController or set ctrl directly. Front faces +x. ``visual_xml`` replaces
    the default box body (put a Poly Pizza car mesh there; chassis bottom is at z = 0.1 in body
    frame when wheels touch the ground, so place meshes with ``pos="0 0 0"`` if they were
    converted with origin="base" and the body origin is at ground level).
    """
    r, wb, tw = wheel_radius, length * 0.6, width * 0.84
    zc = r  # body origin at ground level; wheel centres at z = r
    c = _v(color)
    look = visual_xml or (
        f'<geom type="box" size="{length / 2:.3f} {width / 2:.3f} {0.28:.3f}" pos="0 0 {r + 0.12:.3f}" rgba="{c}" contype="0" conaffinity="0" group="2" mass="0"/>'
        f'<geom type="box" size="{length * 0.26:.3f} {width / 2 * 0.92:.3f} 0.24" pos="{-length * 0.06:.3f} 0 {r + 0.62:.3f}" rgba="{c}" contype="0" conaffinity="0" group="2" mass="0"/>'
        f'<geom type="box" size="{length * 0.245:.3f} {width / 2 * 0.93:.3f} 0.17" pos="{-length * 0.06:.3f} 0 {r + 0.64:.3f}" rgba="0.5 0.65 0.75 0.7" contype="0" conaffinity="0" group="2" mass="0"/>')
    def wheel(tag, x, y, steer):
        wgeom = (f'<geom name="{name}_{tag}_tire" type="cylinder" size="{r} 0.11" euler="90 0 0" rgba="0.08 0.08 0.08 1" mass="18" '
                 f'friction="1.1 0.01 0.001" condim="3" solref="0.01 1"/>'
                 f'<geom type="cylinder" size="{r * 0.6:.3f} 0.112" euler="90 0 0" rgba="0.7 0.7 0.72 1" contype="0" conaffinity="0" group="2" mass="0"/>')
        spin = f'<joint name="{name}_{tag}_spin" type="hinge" axis="0 1 0" damping="0.5" armature="0.5"/>'
        if steer:
            return (f'<body name="{name}_{tag}_knuckle" pos="{x} {y} {zc}"><joint name="{name}_{tag}_steer" type="hinge" axis="0 0 1" '
                    f'range="-{max_steer_deg} {max_steer_deg}" damping="20" armature="2"/><inertial pos="0 0 0" mass="15" diaginertia="0.15 0.15 0.15"/>'
                    f'<body name="{name}_{tag}_wheel">{spin}{wgeom}</body></body>')
        return f'<body name="{name}_{tag}_wheel" pos="{x} {y} {zc}">{spin}{wgeom}</body>'
    q = math.radians(yaw_deg)
    body = (f'<body name="{name}" pos="{_v(pos)}" euler="0 0 {yaw_deg}"><freejoint name="{name}_root"/>'
            f'<inertial pos="0 0 {r + 0.25:.3f}" mass="{mass}" diaginertia="{mass * (width**2 + 0.6**2) / 12:.1f} {mass * (length**2 + 0.6**2) / 12:.1f} {mass * (length**2 + width**2) / 12:.1f}"/>'
            f'{look}<geom name="{name}_chassis" type="box" size="{length / 2 * 0.95:.3f} {width / 2 * 0.95:.3f} 0.2" pos="0 0 {r + 0.2:.3f}" rgba="1 0 0 0.2" group="3" mass="0"/>'
            + wheel("fl", wb / 2, tw / 2, True) + wheel("fr", wb / 2, -tw / 2, True)
            + wheel("rl", -wb / 2, tw / 2, False) + wheel("rr", -wb / 2, -tw / 2, False)
            + f'<site group="4" name="{name}_imu" pos="0 0 {r + 0.3:.3f}"/>'
            + f'<camera name="{name}_driver" pos="{length * 0.1:.2f} 0.35 {r + 0.85:.2f}" xyaxes="0 -1 0 0 0 1" fovy="75"/>'
            + f'<camera name="{name}_chase" pos="{-length * 1.6:.2f} 0 {r + 2.2:.2f}" xyaxes="0 -1 0 0.35 0 0.94" fovy="60"/></body>')
    ms = math.radians(max_steer_deg)
    acts = [f'<position name="{name}_steer_{s}" joint="{name}_f{s[0]}_steer" kp="3000" kv="100" ctrlrange="-{ms:.3f} {ms:.3f}"/>' for s in ("l", "r")]
    acts += [f'<motor name="{name}_drive_{s}" joint="{name}_r{s[0]}_spin" gear="1" ctrlrange="-{max_torque} {max_torque}"/>' for s in ("l", "r")]
    # brakes on all 4 wheels: ctrl in [0, 1]; apply with car_brake_torque() so braking never reverses
    acts += [f'<motor name="{name}_brake_{t}" joint="{name}_{t}_spin" gear="1" ctrlrange="-{max_brake} {max_brake}"/>' for t in ("fl", "fr", "rl", "rr")]
    sens = [f'<velocimeter name="{name}_vel" site="{name}_imu"/>', f'<gyro name="{name}_gyro" site="{name}_imu"/>']
    exc = [f'<exclude body1="{name}" body2="{name}_{t}_wheel"/>' for t in ("fl", "fr", "rl", "rr")]
    exc += [f'<exclude body1="{name}" body2="{name}_{t}_knuckle"/>' for t in ("fl", "fr")]
    return Robot(name, body, acts, sens, exc, home={},
                 info={"wheelbase": wb, "track": tw, "wheel_radius": r, "max_steer": ms, "cameras": [f"{name}_driver", f"{name}_chase"],
                       "note": "qpos order after the free joint: fl_steer, fl_spin, fr_steer, fr_spin, rl_spin, rr_spin"})


def car_brake_torque(model, data, wheel_joint_ids, brake: float, max_brake: float, k: float = 400.0):
    """Brake torques for wheel joints: oppose spin, proportional to speed near standstill (no reverse)."""
    out = []
    for j in wheel_joint_ids:
        w = float(data.qvel[model.jnt_dofadr[j]])
        out.append(-float(np.clip(k * w, -1.0, 1.0)) * brake * max_brake)
    return np.array(out)


class CarController:
    """Speed PI + pure-pursuit steering along ``waypoints`` [(x, y), ...] (loops if ``loop``)."""

    def __init__(self, model, name="car", waypoints=None, speed=8.0, loop=True, lookahead=6.0):
        self.m = model
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_root")
        self.da = model.jnt_dofadr[jid]
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        self.steer = [aid(f"{name}_steer_l"), aid(f"{name}_steer_r")]
        self.drive = [aid(f"{name}_drive_l"), aid(f"{name}_drive_r")]
        self.brakes = [aid(f"{name}_brake_{t}") for t in ("fl", "fr", "rl", "rr")]
        self.brake_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_{t}_spin") for t in ("fl", "fr", "rl", "rr")]
        self.max_brake = float(model.actuator_ctrlrange[self.brakes[0], 1]) if self.brakes[0] >= 0 else 0.0
        self.max_steer = float(model.actuator_ctrlrange[self.steer[0], 1])
        self.tmax = float(model.actuator_ctrlrange[self.drive[0], 1])
        p = model.body_pos[self.bid]
        self.waypoints = np.array(waypoints if waypoints is not None else [(p[0] + 1000, p[1])], float)
        self.speed, self.loop, self.L = speed, loop, lookahead
        self.idx, self.integ = 0, 0.0
        self.wheelbase = 2.7

    def __call__(self, model, data):
        pos = data.xpos[self.bid][:2]
        R = data.xmat[self.bid].reshape(3, 3)
        fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
        v = float(data.qvel[self.da:self.da + 3] @ R[:, 0])
        while np.linalg.norm(self.waypoints[self.idx] - pos) < self.L:
            if self.idx + 1 < len(self.waypoints):
                self.idx += 1
            elif self.loop:
                self.idx = 0
            else:
                break
        tgt = self.waypoints[self.idx]
        d = tgt - pos
        alpha = math.atan2(fwd[0] * d[1] - fwd[1] * d[0], fwd @ d)
        steer = math.atan2(2 * self.wheelbase * math.sin(alpha), max(np.linalg.norm(d), 1e-3))
        data.ctrl[self.steer] = np.clip(steer, -self.max_steer, self.max_steer)
        done = (not self.loop) and self.idx == len(self.waypoints) - 1 and np.linalg.norm(d) < 2.0
        err = (0.0 if done else self.speed) - v
        self.integ = np.clip(self.integ + err * model.opt.timestep, -5, 5)
        tq = 300 * err + 60 * self.integ
        if tq < 0 and self.brakes[0] >= 0:  # slow down with the brakes, never by reversing the motor
            brake = float(np.clip(-tq / self.tmax, 0, 1))
            data.ctrl[self.drive] = 0.0
            data.ctrl[self.brakes] = car_brake_torque(model, data, self.brake_joints, brake, self.max_brake)
            self.integ = max(self.integ, 0.0)
        else:
            data.ctrl[self.drive] = np.clip(tq, 0.0 if v > -0.2 else -self.tmax, self.tmax)
            if self.brakes[0] >= 0:
                data.ctrl[self.brakes] = 0.0


# =========================================================================== diff drive

def diff_drive(name="amr", pos=(0, 0, 0), yaw_deg=0.0, radius=0.3, height=0.25, mass=30.0, wheel_radius=0.08,
               color=(0.9, 0.55, 0.1, 1), top_xml: str = "") -> Robot:
    """Two driven wheels + two casters. Velocity actuators ``{name}_left/right`` in rad/s.
    ``top_xml`` is placed on the deck (z = height) - e.g. a shelf, a lidar, an arm base body."""
    r = wheel_radius
    c = _v(color)
    body = (f'<body name="{name}" pos="{_v(pos)}" euler="0 0 {yaw_deg}"><freejoint name="{name}_root"/>'
            f'<geom name="{name}_base" type="cylinder" size="{radius} {height / 2 - 0.02:.3f}" pos="0 0 {height / 2 + 0.02:.3f}" rgba="{c}" mass="{mass}"/>'
            f'<geom type="cylinder" size="0.05 0.03" pos="{radius * 0.6:.3f} 0 {height + 0.03:.3f}" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2" mass="0"/>'
            f'<geom name="{name}_caster_f" type="sphere" size="0.03" pos="{radius * 0.75:.3f} 0 0.03" friction="0.01 0.001 0.001" rgba="0.3 0.3 0.3 1" mass="0.2"/>'
            f'<geom name="{name}_caster_b" type="sphere" size="0.03" pos="{-radius * 0.75:.3f} 0 0.03" friction="0.01 0.001 0.001" rgba="0.3 0.3 0.3 1" mass="0.2"/>'
            + "".join(f'<body name="{name}_wheel_{s}" pos="0 {sy * (radius + 0.03):.3f} {r}"><joint name="{name}_wheel_{s}" axis="0 1 0" damping="0.05" armature="0.01"/>'
                      f'<geom type="cylinder" size="{r} 0.025" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="1" friction="1.2 0.01 0.001"/></body>'
                      for s, sy in (("l", 1), ("r", -1)))
            + f'<site group="4" name="{name}_imu" pos="0 0 {height / 2:.3f}"/><camera name="{name}_cam" pos="{radius:.2f} 0 {height + 0.1:.2f}" xyaxes="0 -1 0 0 0 1" fovy="70"/>'
            + f'<body name="{name}_deck" pos="0 0 {height:.3f}">{top_xml}</body></body>')
    acts = [f'<velocity name="{name}_{s}" joint="{name}_wheel_{s}" kv="20" ctrlrange="-15 15" forcerange="-40 40"/>' for s in ("l", "r")]
    exc = [f'<exclude body1="{name}" body2="{name}_wheel_{s}"/>' for s in ("l", "r")]
    return Robot(name, body, acts, [f'<velocimeter name="{name}_vel" site="{name}_imu"/>'], exc, home={},
                 info={"wheel_radius": r, "track": 2 * (radius + 0.03), "deck_body": f"{name}_deck", "camera": f"{name}_cam"})


class DiffDriveController:
    def __init__(self, model, name="amr", goal=(2.0, 0.0), v_max=1.0):
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.a = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_{s}") for s in ("l", "r")]
        wl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_wheel_l")
        self.r = float(model.geom_size[model.body_geomadr[wl], 0])
        self.track = 2 * abs(float(model.body_pos[wl][1]))
        self.goal, self.v_max = np.array(goal, float), v_max

    def __call__(self, model, data):
        p = data.xpos[self.bid][:2]
        R = data.xmat[self.bid].reshape(3, 3)
        d = self.goal - p
        dist = float(np.linalg.norm(d))
        head = math.atan2(R[1, 0], R[0, 0])
        err = (math.atan2(d[1], d[0]) - head + math.pi) % (2 * math.pi) - math.pi
        v = 0.0 if dist < 0.05 else min(self.v_max, 0.8 * dist) * max(0.0, math.cos(err))
        w = 2.5 * err if dist > 0.05 else 0.0
        data.ctrl[self.a[0]] = (v - w * self.track / 2) / self.r
        data.ctrl[self.a[1]] = (v + w * self.track / 2) / self.r


# =========================================================================== arm

def arm(name="arm", pos=(0, 0, 0), dof=6, reach=0.85, gripper=True, color=(0.85, 0.87, 0.9, 1), accent=(0.95, 0.45, 0.1, 1),
        payload=2.0) -> Robot:
    """UR-style arm fixed at ``pos`` (base on the surface below). Joints j1..jN, position actuators
    with gains scaled to link inertia, gravcomp on links, parallel gripper (0 = open, 0.04 = closed
    per finger). End-effector site ``{name}_ee`` sits between the fingertips."""
    assert dof in (4, 5, 6, 7)
    c, ac = _v(color), _v(accent)
    L1, L2, L3 = 0.16, reach * 0.47, reach * 0.42
    wr = reach * 0.11
    seq = [("z", L1, 0.065), ("y", L2, 0.055), ("y", L3, 0.045), ("y", wr, 0.04), ("z", wr * 0.8, 0.035), ("x", 0.06, 0.035), ("z", 0.05, 0.03)][:dof]
    axes = {"z": "0 0 1", "y": "0 1 0", "x": "1 0 0"}
    kp = [800, 800, 500, 200, 120, 80, 60][:dof]
    names = []

    def link(i):
        ax, length, rad = seq[i]
        jn = f"{name}_j{i + 1}"
        names.append(jn)
        rng = "-175 175" if ax in "zx" else ("-120 120" if i > 1 else "-100 100")
        if i == 0:
            geom = f'<geom type="cylinder" size="{rad} {length / 2:.3f}" pos="0 0 {length / 2:.3f}" rgba="{c}" mass="3"/>'
            child = f'pos="0 0 {length:.3f}"'
        elif i in (1, 2):
            geom = (f'<geom type="capsule" fromto="0 0 0 0 0 {length:.3f}" size="{rad}" rgba="{c}" mass="{3.0 if i == 1 else 2.0}"/>'
                    f'<geom type="cylinder" size="{rad * 1.15:.3f} {rad * 1.1:.3f}" euler="90 0 0" rgba="{ac}" contype="0" conaffinity="0" group="2" mass="0"/>')
            child = f'pos="0 0 {length:.3f}"'
        else:
            geom = f'<geom type="capsule" fromto="0 0 0 0 0 {length:.3f}" size="{rad}" rgba="{c if i % 2 else ac}" mass="0.6"/>'
            child = f'pos="0 0 {length:.3f}"'
        joint = f'<joint name="{jn}" axis="{axes[ax]}" range="{rng}" damping="{2.0 if i < 3 else 0.5}" armature="{0.1 if i < 3 else 0.02}"/>'
        inner = link(i + 1) if i + 1 < dof else hand()
        return f'<body name="{name}_l{i + 1}" gravcomp="1">{joint}{geom}<body name="{name}_l{i + 1}_end" {child}>{inner}</body></body>'

    def hand():
        if not gripper:
            return f'<site group="4" name="{name}_ee" pos="0 0 0.02" size="0.01"/>'
        fing = "".join(
            f'<body name="{name}_finger_{s}" pos="0 {sy * 0.045:.3f} 0.05" gravcomp="1"><joint name="{name}_finger_{s}" type="slide" axis="0 {-sy} 0" range="0 0.04" damping="5" armature="0.01"/>'
            f'<geom name="{name}_pad_{s}" type="box" size="0.012 0.008 0.035" pos="0 0 0.035" rgba="0.2 0.2 0.22 1" mass="0.05" friction="1.5 0.02 0.002" condim="4"/></body>'
            for s, sy in (("l", 1), ("r", -1)))
        return (f'<body name="{name}_hand" gravcomp="1"><geom type="box" size="0.03 0.06 0.025" pos="0 0 0.025" rgba="0.25 0.25 0.28 1" mass="0.4"/>'
                f'{fing}<site group="4" name="{name}_ee" pos="0 0 0.115" size="0.01" rgba="1 0 0 1"/>'
                # wrist camera beside the palm, looking past the fingertips (hand +z) at what is being grasped
                f'<camera name="{name}_wrist_cam" pos="-0.055 0 0.035" xyaxes="{_lookat((-0.055, 0, 0.035), (0, 0, 0.2), (1, 0, 0))}" fovy="75"/></body>')

    chain = link(0)
    body = (f'<body name="{name}" pos="{_v(pos)}"><geom type="cylinder" size="0.09 0.02" pos="0 0 0.02" rgba="0.2 0.2 0.22 1" mass="4"/>'
            f'<body name="{name}_mount" pos="0 0 0.04">{chain}</body></body>')
    acts = [f'<position name="{name}_a{i + 1}" joint="{jn}" kp="{kp[i]}" kv="{kp[i] * 0.08:.1f}" ctrlrange="{-3.05 if seq[i][0] in "zx" else -2.0} {3.05 if seq[i][0] in "zx" else 2.0}" forcerange="-{kp[i] * 0.4:.0f} {kp[i] * 0.4:.0f}"/>'
            for i, jn in enumerate(names)]
    exc = [f'<exclude body1="{name}" body2="{name}_l1"/>', f'<exclude body1="{name}_mount" body2="{name}_l1"/>']
    for i in range(1, dof):
        exc.append(f'<exclude body1="{name}_l{i}" body2="{name}_l{i + 1}"/>')
        exc.append(f'<exclude body1="{name}_l{i}_end" body2="{name}_l{i + 1}"/>')
    if gripper:
        acts.append(f'<position name="{name}_grip_l" joint="{name}_finger_l" kp="300" kv="10" ctrlrange="0 0.04" forcerange="-40 40"/>')
        acts.append(f'<position name="{name}_grip_r" joint="{name}_finger_r" kp="300" kv="10" ctrlrange="0 0.04" forcerange="-40 40"/>')
        exc += [f'<exclude body1="{name}_l{dof}_end" body2="{name}_hand"/>', f'<exclude body1="{name}_l{dof}" body2="{name}_hand"/>']
    home = [0.0, 0.35, 1.35, 1.44, 0.0, 0.0, 0.0][:dof]  # leans forward (+x), gripper pointing down
    home_d = {jn: v for jn, v in zip(names, home)}
    return Robot(name, body, acts, [f'<framepos name="{name}_ee_pos" objtype="site" objname="{name}_ee"/>'], exc, home=home_d,
                 info={"joints": names, "home": home, "ee_site": f"{name}_ee", "reach": reach,
                       "wrist_camera": f"{name}_wrist_cam" if gripper else None,
                       "gripper": "ctrl 0 = open, 0.04 = closed (both finger actuators)"})


class ArmController:
    """Damped-least-squares IK to ``.target`` (xyz, optional ``.target_down`` keeps the gripper
    pointing down) + gripper command ``.grip`` in [0, 1]. Writes position-actuator targets."""

    def __init__(self, model, name="arm", damping=0.05, max_step=0.05):
        self.m = model
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_ee")
        self.joints, self.acts = [], []
        i = 1
        while True:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_j{i}")
            if j < 0:
                break
            self.joints.append(j)
            self.acts.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_a{i}"))
            i += 1
        self.dofs = [model.jnt_dofadr[j] for j in self.joints]
        self.qadr = [model.jnt_qposadr[j] for j in self.joints]
        self.grip_acts = [a for a in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_grip_{s}") for s in "lr") if a >= 0]
        self.damping, self.max_step = damping, max_step
        self.target, self.target_down, self.grip = None, True, 0.0
        self.q_cmd = None

    def __call__(self, model, data):
        q = np.array([data.qpos[a] for a in self.qadr])
        if self.q_cmd is None:
            self.q_cmd = q.copy()
        if self.target is not None:
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, self.site)
            err = np.asarray(self.target, float) - data.site_xpos[self.site]
            n = np.linalg.norm(err)
            if n > self.max_step:
                err *= self.max_step / n
            J, e = jp[:, self.dofs], err
            if self.target_down:
                z = data.site_xmat[self.site].reshape(3, 3)[:, 2]
                rot_err = np.cross(z, np.array([0, 0, -1.0]))
                J = np.vstack([J, 0.5 * jr[:, self.dofs]])
                e = np.concatenate([err, 0.5 * rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(len(e)), e)
            self.q_cmd = q + dq
        lo = model.actuator_ctrlrange[self.acts, 0]
        hi = model.actuator_ctrlrange[self.acts, 1]
        data.ctrl[self.acts] = np.clip(self.q_cmd, lo, hi)
        for a in self.grip_acts:
            data.ctrl[a] = 0.04 * float(np.clip(self.grip, 0, 1))


# =========================================================================== quadruped

def quadruped(name="dog", pos=(0, 0, 0.0), mass=12.0, body_len=0.6, body_w=0.25, leg=0.25, color=(0.95, 0.75, 0.15, 1)) -> Robot:
    """12-DoF quadruped (hip abduction x, hip pitch y, knee y per leg: fl, fr, rl, rr).
    Standing keyframe: hip 0, thigh 0.75, knee -1.5 rad. ``pos`` z is ignored (stand height set)."""
    c = _v(color)
    hx, hy = body_len / 2 - 0.05, body_w / 2
    q_th, q_kn = 0.75, -1.5
    stand_h = leg * math.cos(q_th) + leg * math.cos(q_th + q_kn) + 0.022
    legs = []
    for tag, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)):
        legs.append(
            f'<body name="{name}_{tag}_hip" pos="{sx * hx:.3f} {sy * hy:.3f} 0">'
            f'<joint name="{name}_{tag}_abd" axis="1 0 0" range="-45 45" damping="0.5" armature="0.02"/>'
            f'<geom type="cylinder" size="0.045 0.03" euler="90 0 0" rgba="0.15 0.15 0.15 1" mass="0.6"/>'
            f'<body name="{name}_{tag}_thigh" pos="0 {sy * 0.07:.3f} 0">'
            f'<joint name="{name}_{tag}_hip" axis="0 1 0" range="-60 140" damping="0.5" armature="0.02"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 -{leg}" size="0.025" rgba="{c}" mass="1.0"/>'
            f'<body name="{name}_{tag}_shin" pos="0 0 -{leg}">'
            f'<joint name="{name}_{tag}_knee" axis="0 1 0" range="-160 -30" damping="0.5" armature="0.02"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 -{leg}" size="0.018" rgba="0.15 0.15 0.15 1" mass="0.3"/>'
            f'<geom name="{name}_{tag}_foot" type="sphere" size="0.022" pos="0 0 -{leg}" rgba="0.1 0.1 0.1 1" mass="0.05" friction="1.2 0.02 0.002" condim="3"/>'
            f'<site group="4" name="{name}_{tag}_foot_site" pos="0 0 -{leg}" size="0.025"/></body></body></body>')
    trunk_m = mass - 4 * (0.6 + 1.0 + 0.35)
    body = (f'<body name="{name}" pos="{pos[0]} {pos[1]} {stand_h:.3f}"><freejoint name="{name}_root"/>'
            f'<geom name="{name}_trunk" type="box" size="{body_len / 2} {body_w / 2} 0.06" rgba="{c}" mass="{trunk_m:.2f}"/>'
            f'<geom type="box" size="0.05 0.06 0.04" pos="{body_len / 2 + 0.03} 0 0.02" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" group="2" mass="0"/>'
            f'<site group="4" name="{name}_imu"/><camera name="{name}_head_cam" pos="{body_len / 2 + 0.08} 0 0.04" xyaxes="0 -1 0 0 0 1" fovy="80"/>'
            + "".join(legs) + "</body>")
    acts, joints = [], []
    for tag in ("fl", "fr", "rl", "rr"):
        for jn, kp in (("abd", 60), ("hip", 60), ("knee", 60)):
            joints.append(f"{name}_{tag}_{jn}")
            acts.append(f'<position name="{name}_{tag}_{jn}_act" joint="{name}_{tag}_{jn}" kp="{kp}" kv="2" ctrlrange="-3 3" forcerange="-35 35"/>')
    sens = [f'<gyro name="{name}_gyro" site="{name}_imu"/>', f'<accelerometer name="{name}_acc" site="{name}_imu"/>',
            f'<framequat name="{name}_quat" objtype="site" objname="{name}_imu"/>']
    sens += [f'<touch name="{name}_{t}_touch" site="{name}_{t}_foot_site"/>' for t in ("fl", "fr", "rl", "rr")]
    stand = [0.0, q_th, q_kn] * 4
    return Robot(name, body, acts, sens, home=dict(zip(joints, stand)),
                 info={"joints": joints, "stand_pose": stand, "stand_height": stand_h, "camera": f"{name}_head_cam"})


class StandController:
    """Hold the "home" keyframe joint pose (any robot with position actuators on hinge/slide joints).
    Optional trot: ``.gait = True`` adds a small open-loop diagonal-pair stepping motion."""

    def __init__(self, model, key=0, gait=False, freq=2.0, amp=0.25):
        self.target = np.zeros(model.nu)
        self.jq = []
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            self.jq.append(model.jnt_qposadr[j])
            self.target[a] = model.key_qpos[key, model.jnt_qposadr[j]] if model.nkey else model.qpos0[model.jnt_qposadr[j]]
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "" for a in range(model.nu)]
        self.gait, self.freq, self.amp = gait, freq, amp

    def __call__(self, model, data):
        u = self.target.copy()
        if self.gait:
            ph = 2 * math.pi * self.freq * data.time
            for a, n in enumerate(self.names):
                diag = ("_fl_" in n or "_rr_" in n)
                s = math.sin(ph + (0 if diag else math.pi))
                if n.endswith("_hip_act"):
                    u[a] += 0.5 * self.amp * s
                elif n.endswith("_knee_act"):
                    u[a] -= self.amp * max(0.0, s)
        data.ctrl[:] = u
