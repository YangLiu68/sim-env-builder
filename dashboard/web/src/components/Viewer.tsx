import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "../api";

type Scene = any;

function b64ToArray<T extends Float32Array | Int32Array>(b64: string, Ctor: { new (buf: ArrayBuffer): T }): T {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Ctor(buf);
}

const col = (rgba: number[]) => new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);

function buildGeometry(g: any, sc: Scene, cache: Map<string, THREE.BufferGeometry>): THREE.BufferGeometry | null {
  const s = g.size;
  switch (g.type) {
    case "plane": {
      const sx = s[0] > 0 ? s[0] : 250, sy = s[1] > 0 ? s[1] : 250;
      return new THREE.PlaneGeometry(2 * sx, 2 * sy, 1, 1);
    }
    case "box": return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "sphere": return new THREE.SphereGeometry(s[0], 28, 18);
    case "ellipsoid": { const geo = new THREE.SphereGeometry(1, 28, 18); geo.scale(s[0], s[1], s[2]); return geo; }
    case "capsule": { const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20); geo.rotateX(Math.PI / 2); return geo; }
    case "cylinder": { const geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 28); geo.rotateX(Math.PI / 2); return geo; }
    case "mesh": {
      const key = `m${g.mesh}`;
      if (!cache.has(key)) {
        const m = sc.meshes[String(g.mesh)];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(b64ToArray(m.vert, Float32Array), 3));
        geo.setIndex(new THREE.BufferAttribute(new Uint32Array(b64ToArray(m.face, Int32Array).buffer), 1));
        const ni = geo.toNonIndexed();
        ni.computeVertexNormals();
        cache.set(key, ni);
      }
      return cache.get(key)!;
    }
    case "hfield": {
      const h = sc.hfields[String(g.hfield)];
      const [sx, sy, sz] = h.size;
      const geo = new THREE.PlaneGeometry(2 * sx, 2 * sy, h.ncol - 1, h.nrow - 1);
      const data = b64ToArray(h.data, Float32Array);
      const pos = geo.attributes.position as THREE.BufferAttribute;
      for (let iy = 0; iy < h.nrow; iy++) {
        for (let ix = 0; ix < h.ncol; ix++) {
          const r = h.nrow - 1 - iy; // PlaneGeometry rows go from +y to -y; MuJoCo row 0 is -y
          pos.setZ(iy * h.ncol + ix, data[r * h.ncol + ix] * sz);
        }
      }
      geo.computeVertexNormals();
      return geo;
    }
  }
  return null;
}

export default function Viewer({ runId }: { runId: string }) {
  const host = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<any>({});
  const [hud, setHud] = useState({ t: 0, rtf: 0, ncon: 0 });
  const [camMode, setCamMode] = useState("orbit");
  const [showCol, setShowCol] = useState(false);
  const [shadows, setShadows] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);
  const [ctrl, setCtrl] = useState<number[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const ctx = useRef<any>({});

  useEffect(() => {
    setScene(null); setErr(null);
    api.scene(runId).then(setScene).catch((e) => setErr(String(e.message || e)));
  }, [runId]);

  const robots = useMemo(() => {
    if (!scene) return [];
    const roots = new Set<number>();
    const top = (b: number) => { while (b && scene.bodies[b].parent !== 0) b = scene.bodies[b].parent; return b; };
    for (const g of scene.geoms) { /* bodies with actuated subtrees are found via actuator names below */ void g; }
    scene.bodies.forEach((b: any) => { if (b.id && b.parent === 0) roots.add(b.id); });
    // prefer bodies whose names prefix actuator names
    const acts = scene.actuators.map((a: any) => a.name);
    const scored = [...roots].map((r) => ({ id: r, name: scene.bodies[r].name, hit: acts.some((n: string) => n.startsWith(scene.bodies[r].name)) }));
    return scored.filter((x) => x.hit).concat(scored.filter((x) => !x.hit)).slice(0, 30).map((x) => ({ ...x, top: top(x.id) }));
  }, [scene]);

  // ---------- three.js setup ----------
  useEffect(() => {
    if (!scene || !host.current) return;
    const el = host.current;
    const W = () => el.clientWidth, H = () => el.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W(), H());
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    el.appendChild(renderer.domElement);
    const top = scene.sky.top, hor = scene.sky.horizon;
    const css = (c: number[]) => `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
    el.style.background = `linear-gradient(180deg, ${css(top)} 0%, ${css(hor)} 70%, ${css(hor)} 100%)`;

    const three = new THREE.Scene();
    const [lo, hi] = scene.bounds;
    const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    const span = Math.max(1, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
    three.fog = new THREE.Fog(new THREE.Color().setRGB(hor[0], hor[1], hor[2], THREE.SRGBColorSpace), span * 3, span * 12);
    const camera = new THREE.PerspectiveCamera(45, W() / H(), 0.01, span * 40);
    camera.up.set(0, 0, 1);
    camera.position.set(center.x + span * 0.75, center.y - span * 0.75, center.z + span * 0.55);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    three.add(new THREE.HemisphereLight(0xdfe9ff, 0x3b3a33, 0.9));
    const sun = new THREE.DirectionalLight(0xfff4e2, 2.2);
    sun.position.set(center.x - span * 0.6, center.y - span * 0.4, center.z + span * 1.4);
    sun.target.position.copy(center);
    sun.castShadow = true;
    const sh = span * 0.75;
    Object.assign(sun.shadow.camera, { left: -sh, right: sh, top: sh, bottom: -sh, near: 0.1, far: span * 4 });
    sun.shadow.mapSize.set(4096, 4096);
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    three.add(sun, sun.target);

    const texLoader = new THREE.TextureLoader();
    const textures = new Map<number, THREE.Texture>();
    const geoCache = new Map<string, THREE.BufferGeometry>();
    const bodyGroups: THREE.Group[] = scene.bodies.map(() => new THREE.Group());
    bodyGroups.forEach((g) => three.add(g));
    const colMeshes: THREE.Mesh[] = [];
    const pickables: THREE.Object3D[] = [];
    for (const g of scene.geoms) {
      const geo = buildGeometry(g, scene, geoCache);
      if (!geo) continue;
      const transparent = g.rgba[3] < 0.999;
      const isCollisionProxy = g.group >= 3;
      const mat = new THREE.MeshStandardMaterial({
        color: col(g.rgba), transparent: transparent || isCollisionProxy, opacity: isCollisionProxy ? 0.35 : g.rgba[3],
        roughness: g.type === "plane" ? 0.95 : 0.72, metalness: 0.04, flatShading: g.type === "mesh",
        depthWrite: !(transparent || isCollisionProxy),
      });
      if (g.texture != null && scene.textures[String(g.texture)]) {
        // one texture object per (texture, repeat): data URLs decode once in the browser cache
        const rep = g.texrepeat || [1, 1];
        const key = g.texture * 100003 + Math.round(rep[0] * 97 + rep[1] * 13);
        if (!textures.has(key)) {
          const t = texLoader.load(scene.textures[String(g.texture)]);
          t.wrapS = t.wrapT = THREE.RepeatWrapping;
          t.colorSpace = THREE.SRGBColorSpace;
          t.anisotropy = 8;
          t.repeat.set(rep[0], rep[1]);
          textures.set(key, t);
        }
        mat.map = textures.get(key)!;
        mat.color.set(0xffffff);
      }
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
      mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      mesh.castShadow = g.type !== "plane" && g.type !== "hfield" && !transparent;
      mesh.receiveShadow = true;
      mesh.userData = { body: g.body, geom: g.id };
      if (isCollisionProxy) { mesh.visible = false; colMeshes.push(mesh); }
      else if (g.rgba[3] === 0) mesh.visible = false;
      bodyGroups[g.body].add(mesh);
      if (g.body !== 0) pickables.push(mesh);
    }
    // initial poses from bodies are applied by the first frame; until then world-only geoms are fine
    const raycaster = new THREE.Raycaster();
    const onClick = (ev: MouseEvent) => {
      if (!ev.shiftKey || !ws.current) return;
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster.intersectObjects(pickables, false)[0];
      if (!hit) return;
      const dir = raycaster.ray.direction.clone();
      dir.z = Math.max(dir.z, 0.15);
      dir.normalize();
      ws.current.send(JSON.stringify({ cmd: "push", body: hit.object.userData.body, dir: [dir.x, dir.y, dir.z] }));
    };
    renderer.domElement.addEventListener("click", onClick);
    const ro = new ResizeObserver(() => { renderer.setSize(W(), H()); camera.aspect = W() / H(); camera.updateProjectionMatrix(); });
    ro.observe(el);

    ctx.current = { renderer, three, camera, controls, bodyGroups, colMeshes, sun, frame: null as Float32Array | null, follow: -1, camIdx: -1, prevFollow: new THREE.Vector3() };
    let raf = 0;
    const nb = scene.nbody, nc = scene.cameras.length;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const c = ctx.current;
      const f: Float32Array | null = c.frame;
      if (f) {
        const xp = 3, xq = 3 + nb * 3, cp = xq + nb * 4, cm = cp + nc * 3;
        for (let b = 1; b < nb; b++) {
          const grp = bodyGroups[b];
          grp.position.set(f[xp + 3 * b], f[xp + 3 * b + 1], f[xp + 3 * b + 2]);
          grp.quaternion.set(f[xq + 4 * b + 1], f[xq + 4 * b + 2], f[xq + 4 * b + 3], f[xq + 4 * b]);
        }
        if (c.follow > 0) {
          const p = new THREE.Vector3(f[xp + 3 * c.follow], f[xp + 3 * c.follow + 1], f[xp + 3 * c.follow + 2]);
          const delta = p.clone().sub(c.prevFollow);
          if (c.prevFollow.lengthSq() > 0) { camera.position.add(delta); controls.target.add(delta); }
          else controls.target.copy(p);
          c.prevFollow.copy(p);
        }
        if (c.camIdx >= 0) {
          const i = c.camIdx;
          camera.position.set(f[cp + 3 * i], f[cp + 3 * i + 1], f[cp + 3 * i + 2]);
          const m = f.subarray(cm + 9 * i, cm + 9 * i + 9);
          const rot = new THREE.Matrix4().set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
          camera.quaternion.setFromRotationMatrix(rot);
        }
      }
      if (c.camIdx < 0) controls.update();
      renderer.render(three, camera);
    };
    loop();

    // ---------- websocket ----------
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/sim`);
    sock.binaryType = "arraybuffer";
    sock.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const m = JSON.parse(ev.data);
        if (m.type === "error") setErr(m.error);
        else { setInfo(m); if (m.ctrl) setCtrl((old) => (old.length ? old : m.ctrl)); }
        return;
      }
      const f = new Float32Array(ev.data);
      ctx.current.frame = f;
      setHud((h) => (Math.abs(h.t - f[0]) > 0.05 ? { t: f[0], rtf: f[1], ncon: f[2] } : h));
    };
    sock.onerror = () => setErr("simulation connection failed");
    ws.current = sock;

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("click", onClick);
      sock.close();
      controls.dispose();
      renderer.dispose();
      three.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.map?.dispose?.(); o.material?.dispose?.(); });
      el.removeChild(renderer.domElement);
    };
  }, [scene, runId]);

  useEffect(() => { ctx.current.colMeshes?.forEach((m: THREE.Mesh) => (m.visible = showCol)); }, [showCol, scene]);
  useEffect(() => {
    const c = ctx.current;
    if (!c.renderer) return;
    c.renderer.shadowMap.enabled = shadows;
    c.sun.castShadow = shadows;
    c.three.traverse((o: any) => { if (o.material) o.material.needsUpdate = true; });
  }, [shadows, scene]);
  useEffect(() => {
    const c = ctx.current;
    if (!c.camera || !scene) return;
    c.follow = -1; c.camIdx = -1; c.prevFollow = new THREE.Vector3();
    c.controls.enabled = true;
    if (camMode.startsWith("cam:")) {
      const i = Number(camMode.slice(4));
      c.camIdx = i;
      c.controls.enabled = false;
      c.camera.fov = scene.cameras[i].fovy;
      c.camera.updateProjectionMatrix();
    } else {
      c.camera.fov = 45; c.camera.updateProjectionMatrix();
      if (camMode.startsWith("follow:")) c.follow = Number(camMode.slice(7));
    }
  }, [camMode, scene]);

  const send = (m: any) => ws.current?.readyState === 1 && ws.current.send(JSON.stringify(m));

  if (err) return <div className="viewer"><div className="v-loading"><div style={{ textAlign: "center", maxWidth: 520 }}><div style={{ fontWeight: 600, color: "var(--text)" }}>Viewer unavailable</div><div className="mono" style={{ marginTop: 6 }}>{err}</div></div></div></div>;
  if (!scene) return <div className="viewer"><div className="v-loading">Loading scene…</div></div>;

  return (
    <div className="viewer" ref={host}>
      <div className="v-toolbar">
        <button className="icon-btn" title={info.playing ? "Pause" : "Play"} onClick={() => send({ cmd: info.playing ? "pause" : "play" })}>
          {info.playing ? <svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="1" width="3.5" height="12" rx="1" fill="currentColor" /><rect x="8.5" y="1" width="3.5" height="12" rx="1" fill="currentColor" /></svg>
            : <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1.5v11l9-5.5z" fill="currentColor" /></svg>}
        </button>
        <button className="icon-btn" title="Reset to the home keyframe" onClick={() => { send({ cmd: "reset" }); setCtrl([]); }}>
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="M2.5 8a5.5 5.5 0 1 0 1.7-4" /><path d="M2 2.5v3.5h3.5" /></svg>
        </button>
        <select value={String(info.speed ?? 1)} onChange={(e) => send({ cmd: "speed", value: Number(e.target.value) })} title="Playback speed">
          {[0.25, 0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        <select value={camMode} onChange={(e) => setCamMode(e.target.value)} title="Camera">
          <option value="orbit">Orbit camera</option>
          {robots.slice(0, 12).map((r) => <option key={r.id} value={`follow:${r.id}`}>Follow {r.name}</option>)}
          {scene.cameras.map((c: any, i: number) => <option key={c.id} value={`cam:${i}`}>Camera · {c.name}</option>)}
        </select>
        <button className={`icon-btn ${showCol ? "on" : ""}`} title="Show collision geometry" onClick={() => setShowCol(!showCol)}>
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 1.5l5.5 3v7L8 14.5l-5.5-3v-7z" /><path d="M2.5 4.5L8 7.5l5.5-3M8 7.5v7" /></svg>
        </button>
        <button className={`icon-btn ${shadows ? "on" : ""}`} title="Shadows" onClick={() => setShadows(!shadows)}>
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="3" /><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6L13 13M3 13l1.4-1.4M11.6 4.4L13 3" /></svg>
        </button>
        <a className="icon-btn" href={`/api/runs/${runId}/render?camera=${camMode.startsWith("cam:") ? camMode.slice(4) : -1}`} target="_blank" rel="noreferrer" title="Open MuJoCo's own render of the current state">
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="1.5" y="3.5" width="13" height="10" rx="2" /><circle cx="8" cy="8.5" r="2.6" /><path d="M5.5 3.5l1-2h3l1 2" /></svg>
        </a>
      </div>
      <div className="v-hud">
        <span>t = {hud.t.toFixed(2)} s</span>
        <span title="How much faster than real time the physics could run on this machine">{hud.rtf ? `${hud.rtf.toFixed(0)}× RT headroom` : "…"}</span>
        <span>{Math.round(hud.ncon)} contacts</span>
        {info.controller_error && <span style={{ color: "var(--bad)" }}>controller error</span>}
      </div>
      <div className={`v-panel ${panelOpen ? "" : "collapsed"}`}>
        <div className="vp-head">
          <span>Controls</span>
          <button className="btn ghost sm" onClick={() => setPanelOpen(!panelOpen)}>{panelOpen ? "Hide" : "Show"}</button>
        </div>
        {panelOpen && <div className="vp-body">
          <label className="row" style={{ justifyContent: "space-between", padding: "6px 0 10px", borderBottom: "1px solid var(--line)", marginBottom: 6 }}>
            <span>
              <div style={{ fontWeight: 600, fontSize: 13 }}>Demo controller</div>
              <div className="muted" style={{ fontSize: 12 }}>{info.has_controller ? "controller.py drives the actuators" : "no controller.py in this env"}</div>
            </span>
            <input type="checkbox" checked={!!info.demo && !!info.has_controller} disabled={!info.has_controller}
              onChange={(e) => { send({ cmd: "demo", on: e.target.checked }); if (e.target.checked) setCtrl([]); }} />
          </label>
          {scene.actuators.length === 0 && <div className="muted" style={{ fontSize: 13, padding: "8px 0" }}>This scene has no actuators.</div>}
          {scene.actuators.map((a: any, i: number) => {
            const live = info.ctrl?.[i] ?? a.default;
            const v = info.demo ? live : (ctrl[i] ?? live);
            const step = (a.range[1] - a.range[0]) / 400 || 0.01;
            return (
              <div className="slider" key={a.id}>
                <div className="sl-top"><span className="sl-name" title={a.name}>{a.name}</span><span className="sl-val">{Number(v).toFixed(2)}</span></div>
                <input type="range" min={a.range[0]} max={a.range[1]} step={step} value={v}
                  onChange={(e) => { const nv = Number(e.target.value); const c2 = [...(ctrl.length ? ctrl : info.ctrl || [])]; c2[i] = nv; setCtrl(c2); send({ cmd: "ctrl", index: i, value: nv }); }} />
              </div>
            );
          })}
        </div>}
        {panelOpen && <div className="vp-foot">Drag to orbit · scroll to zoom · <b>Shift+click</b> an object to push it · moving a slider takes manual control</div>}
      </div>
    </div>
  );
}
