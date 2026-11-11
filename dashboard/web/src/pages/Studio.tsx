import { useState } from "react";
import { api, fileUrl, phaseLabel, RunSummary, timeAgo, verdictPill } from "../api";

const EXAMPLES: [string, string, string][] = [
  ["Drone racing park", "aerial", "A drone racing course in a park: a quadcopter, 6 gates at different heights, trees and a few obstacles. Usable for RL, with an FPV camera."],
  ["Kitchen pick & place", "robot-arm", "A robot arm with a gripper on a kitchen counter with a mug, some fruit and a plate to pick and place, plus a wrist camera."],
  ["City intersection", "self-driving", "A four-way city intersection with traffic lights, buildings, trees, parked cars and an ego car I can steer and accelerate."],
  ["Warehouse AMR", "logistics", "A warehouse with pallet racks, pallets and boxes, a parked forklift, and a differential-drive robot that drives down the aisles."],
  ["Quadruped trail", "legged", "Rough hilly terrain with rocks and a short flight of stairs for a quadruped robot that starts standing."],
  ["Mars rover", "space", "A Mars rover on cratered rocky terrain with Mars gravity, a lander in the distance and some boulders."],
  ["Robot soccer", "multi-agent", "A small soccer field with goals, a ball and two teams of two wheeled robots for multi-agent RL."],
];

export function Card({ r, go }: { r: RunSummary; go: (x: string) => void }) {
  const vp = verdictPill(r.verdict);
  return (
    <div className="card env-card" onClick={() => go(`run/${r.id}`)}>
      {r.thumb ? <img className="thumb" src={fileUrl(r.id, r.thumb)} alt="" loading="lazy" /> : <div className="thumb ph">{r.phase === "awaiting_reply" ? "waiting for your answer" : r.phase ? phaseLabel[r.phase] || r.phase : "no render yet"}</div>}
      <div className="meta">
        <div className="t" title={r.title}>{r.title}</div>
        <div className="row" style={{ marginTop: 6, flexWrap: "wrap", gap: 6 }}>
          {r.category && <span className="pill neutral">{r.category}</span>}
          {r.has_env && <span className={`pill ${vp.cls}`}>{vp.label}</span>}
          {r.score != null && <span className="pill info">{r.score}</span>}
          <span className="muted" style={{ fontSize: 12 }}>{timeAgo(r.updated)}</span>
        </div>
      </div>
    </div>
  );
}

export default function Studio({ runs, go, refresh }: { runs: RunSummary[]; go: (x: string) => void; refresh: () => void }) {
  const [prompt, setPrompt] = useState("");
  const [cat, setCat] = useState<string | undefined>();
  const [busy, setBusy] = useState(false);
  const mine = runs.filter((r) => r.kind === "studio");
  const start = async () => {
    setBusy(true);
    try { const { id } = await api.newStudio(prompt, cat); refresh(); go(`run/${id}`); }
    catch (e: any) { alert(e.message); }
    finally { setBusy(false); }
  };
  return (
    <div className="page">
      <div className="card composer">
        <h2>Describe a simulated environment</h2>
        <div className="muted">A fresh agent reads the sim-env-builder playbook, picks MuJoCo or Isaac Sim, confirms the plan with you, pulls models from Poly Pizza (or generates them), builds the scene and smoke-tests it.</div>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="e.g. A harbor with a small boat on the water, a wooden dock, buoys and a lighthouse. The boat should float and have a thruster and rudder."
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && prompt.trim().length > 8) start(); }} />
        <div className="row" style={{ marginTop: 12 }}>
          <div className="chips grow" style={{ marginTop: 0 }}>
            {EXAMPLES.map(([label, c, text]) => <button key={label} className="chip" onClick={() => { setPrompt(text); setCat(c); }}>{label}</button>)}
          </div>
          <button className="btn primary" disabled={busy || prompt.trim().length < 8} onClick={start}>{busy ? "Starting…" : "Build environment"}</button>
        </div>
        <div className="how">
          {[["1 · Confirm", "Backend choice + plan, asks you 2–4 questions"], ["2 · Assets", "Poly Pizza models, procedural fallbacks"], ["3 · Build", "Scene, robot, controller, Gym env"],
            ["4 · Smoke tests", "Render, physics examiner, task checks"], ["5 · Visual check", "Agent inspects its own renders"]].map(([b, s]) => <div className="step" key={b}><b>{b}</b><span>{s}</span></div>)}
        </div>
      </div>
      <div className="page-head" style={{ marginTop: 28 }}>
        <div><h2 className="h1" style={{ fontSize: 18 }}>Your runs</h2><div className="sub">Environments started from this page</div></div>
      </div>
      {mine.length === 0 ? <div className="card empty">Nothing yet. Describe an environment above, or browse the golden set.</div>
        : <div className="grid-cards">{mine.map((r) => <Card key={r.id} r={r} go={go} />)}</div>}
    </div>
  );
}
