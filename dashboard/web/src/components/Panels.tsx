import { useState } from "react";
import { fileUrl, verdictPill } from "../api";
import Markdown from "./Markdown";

export function Lightbox({ src, onClose }: { src: string | null; onClose: () => void }) {
  if (!src) return null;
  return <div className="lightbox" onClick={onClose}><img src={src} alt="" /></div>;
}

function Img({ runId, path, onOpen, alt }: { runId: string; path?: string | null; onOpen: (s: string) => void; alt: string }) {
  if (!path) return null;
  const src = fileUrl(runId, path);
  return <div className="img-frame" onClick={() => onOpen(src)}><img src={src} alt={alt} loading="lazy" /></div>;
}

export function ResultsPanel({ runId, run }: { runId: string; run: any }) {
  const [open, setOpen] = useState<string | null>(null);
  const env = run.env;
  if (!env) return <div className="empty">No environment yet. Results appear here once the agent has built and smoke-tested the scene.</div>;
  const s = env.physics?.summary || {};
  const vp = verdictPill(env.smoke?.verdict);
  const m = env.manifest || {};
  return (
    <div className="pane">
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <span className={`pill ${vp.cls}`}>{vp.label}</span>
        <span className="pill good">{s.pass ?? 0} pass</span>
        <span className={`pill ${s.warn ? "warn" : "neutral"}`}>{s.warn ?? 0} warn</span>
        <span className={`pill ${s.fail ? "bad" : "neutral"}`}>{s.fail ?? 0} fail</span>
        <span className={`pill ${env.render?.ok ? "good" : "bad"}`}>render {env.render?.ok ? "ok" : "problems"}</span>
        <span className="muted" style={{ fontSize: 12.5 }}>{m.bodies} bodies · {m.geoms} geoms · {m.actuators} actuators · dt {m.timestep}s · g {m.gravity?.[2]}</span>
      </div>
      {env.render?.problems?.length > 0 && <div className="card card-pad" style={{ borderColor: "rgba(242,184,75,.35)" }}>
        <div className="card-title">Render problems</div>
        <ul style={{ margin: 0, paddingLeft: 18 }}>{env.render.problems.map((p: string) => <li key={p} className="muted">{p}</li>)}</ul>
      </div>}
      {env.showcase && <div className="card card-pad"><div className="card-title">Showcase video <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>fly-around · demo controller · every camera</span></div>
        <video src={fileUrl(runId, env.showcase)} controls autoPlay muted loop playsInline style={{ width: "100%", borderRadius: 10, background: "#000" }} /></div>}
      <div className="card card-pad"><div className="card-title">Contact sheet <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>named cameras + automatic views</span></div>
        <Img runId={runId} path={env.contact_sheet} onOpen={setOpen} alt="contact sheet" /></div>
      <div className="card card-pad"><div className="card-title">Rollout filmstrip <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>demo controller, tracking + fixed view</span></div>
        <Img runId={runId} path={env.filmstrip} onOpen={setOpen} alt="filmstrip" />
        {env.video && <video src={fileUrl(runId, env.video)} controls style={{ width: "100%", marginTop: 10, borderRadius: 10 }} />}
      </div>
      <div className="card card-pad"><div className="card-title">Visual vs collision geometry</div>
        <Img runId={runId} path={env.collision_sheet} onOpen={setOpen} alt="collision sheet" /></div>
      {env.renders?.length > 0 && <div className="card card-pad"><div className="card-title">All renders</div>
        <div className="renders">{env.renders.map((r: string) => (
          <figure key={r}><Img runId={runId} path={r} onOpen={setOpen} alt={r} /><figcaption>{r.split("/").pop()}</figcaption></figure>
        ))}</div></div>}
      <Lightbox src={open} onClose={() => setOpen(null)} />
    </div>
  );
}

const statusCls: Record<string, string> = { pass: "good", warn: "warn", fail: "bad", info: "info" };

export function PhysicsPanel({ run }: { run: any }) {
  const checks = run.env?.physics?.checks || [];
  if (!checks.length) return <div className="empty">No physics report yet.</div>;
  const groups: Record<string, any[]> = {};
  checks.forEach((c: any) => (groups[c.group] = groups[c.group] || []).push(c));
  const order = ["model", "initial", "rollout", "probes", "numerics", "actuators", "layout", "task"];
  return (
    <div className="pane">
      <div className="card" style={{ overflow: "hidden" }}>
        <table className="t">
          <thead><tr><th style={{ width: 80 }}>Status</th><th style={{ width: 200 }}>Check</th><th>Value</th><th>Details</th></tr></thead>
          <tbody>
            {Object.keys(groups).sort((a, b) => order.indexOf(a) - order.indexOf(b)).map((g) => [
              <tr className="group-h" key={`h-${g}`}><td colSpan={4}>{g}</td></tr>,
              ...groups[g].map((c, i) => (
                <tr key={`${g}-${i}`}>
                  <td><span className={`pill ${statusCls[c.status] || "neutral"}`}>{c.status}</span></td>
                  <td style={{ fontWeight: 500 }}>{c.name.replace(/_/g, " ")}</td>
                  <td className="val">{c.value == null ? "" : typeof c.value === "object" ? JSON.stringify(c.value) : String(c.value)}{c.expect != null && <div className="muted">expect {String(c.expect)}</div>}</td>
                  <td className="muted" style={{ fontSize: 12.5, maxWidth: 380 }}>{c.details}</td>
                </tr>
              )),
            ])}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function AssetsPanel({ run }: { run: any }) {
  const env = run.env;
  const pp = env?.polypizza || {};
  const entries = Object.entries(pp);
  return (
    <div className="pane">
      <div className="card card-pad">
        <div className="card-title">From Poly Pizza <span className="pill info">{entries.length}</span></div>
        {entries.length === 0 && <div className="muted">No Poly Pizza models in this environment (everything procedural).</div>}
        <div className="grid-cards" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))" }}>
          {entries.map(([role, m]: any) => (
            <a key={role} className="card env-card" href={m.page_url} target="_blank" rel="noreferrer" style={{ textDecoration: "none", color: "inherit" }}>
              {m.preview_url ? <img className="thumb" src={m.preview_url} alt="" /> : <div className="thumb ph">no preview</div>}
              <div className="meta">
                <div className="t">{role}</div>
                <div className="muted" style={{ fontSize: 12 }}>{m.title} · {m.creator}</div>
                <div className="row" style={{ marginTop: 6 }}><span className={`pill ${m.requires_attribution ? "warn" : "good"}`}>{m.licence}</span><span className="muted mono" style={{ fontSize: 11 }}>{m.triangles} tris</span></div>
              </div>
            </a>
          ))}
        </div>
      </div>
      {env?.docs?.["assets/ASSETS.md"] && <div className="card card-pad"><div className="card-title">Asset sources (ASSETS.md)</div><Markdown text={env.docs["assets/ASSETS.md"]} /></div>}
      {env?.docs?.["assets/polypizza/CREDITS.md"] && <div className="card card-pad"><div className="card-title">Credits</div><Markdown text={env.docs["assets/polypizza/CREDITS.md"]} /></div>}
    </div>
  );
}

export function DocsPanel({ run }: { run: any }) {
  const docs = run.env?.docs || {};
  const names = ["SPEC.md", "README.md", "smoke/visual_check.md", "smoke/SMOKE.md"].filter((n) => docs[n]);
  const [sel, setSel] = useState(names[0]);
  if (!names.length) return <div className="empty">No documents yet.</div>;
  const cur = sel && docs[sel] ? sel : names[0];
  return (
    <div className="pane">
      <div className="row">{names.map((n) => <button key={n} className={`btn sm ${n === cur ? "primary" : ""}`} onClick={() => setSel(n)}>{n}</button>)}</div>
      <div className="card card-pad"><Markdown text={docs[cur]} /></div>
    </div>
  );
}

export function JudgePanel({ run, onEvaluate }: { run: any; onEvaluate?: () => void }) {
  const j = run.judge;
  const g = run.grade;
  if (!j && !g) return (
    <div className="empty">
      <div>Not evaluated yet.</div>
      {onEvaluate && <button className="btn primary" style={{ marginTop: 12 }} onClick={onEvaluate}>Run grader + independent judge</button>}
    </div>
  );
  const scores = j?.scores || {};
  return (
    <div className="pane">
      <div className="kpis">
        <div className="card kpi"><div className="k-label">Total</div><div className="k-value">{run.result?.total_score ?? "–"}</div><div className="k-note">out of 100</div></div>
        <div className="card kpi"><div className="k-label">Automated</div><div className="k-value">{g?.auto_score ?? "–"}</div><div className="k-note">out of 40</div></div>
        <div className="card kpi"><div className="k-label">Judge</div><div className="k-value">{j?.judge_score ?? "–"}</div><div className="k-note">out of 60</div></div>
        <div className="card kpi"><div className="k-label">Independent smoke re-run</div><div className="k-value" style={{ fontSize: 18, marginTop: 8 }}>{g?.smoke_regrade?.verdict ?? "–"}</div><div className="k-note">rebuild {g?.rebuild?.ok ? "ok" : "failed"} · env.py {g?.env_py?.ok ? "ok" : "failed"}</div></div>
      </div>
      {j && <div className="card card-pad">
        <div className="card-title">Judge scores</div>
        <div className="score-bars">{Object.entries(scores).map(([k, v]: any) => (
          <div className="score-bar" key={k}><span>{k}</span><div className="track"><div className="fill" style={{ width: `${v * 10}%` }} /></div><span className="mono">{v}/10</span></div>
        ))}</div>
        {j.summary && <p className="muted" style={{ marginBottom: 0 }}>{j.summary}</p>}
      </div>}
      {j?.must_have_missing?.length > 0 && <div className="card card-pad"><div className="card-title">Missing must-haves</div>
        <div className="row" style={{ flexWrap: "wrap" }}>{j.must_have_missing.map((m: string) => <span key={m} className="pill bad">{m}</span>)}</div></div>}
      {j?.problems?.length > 0 && <div className="card card-pad"><div className="card-title">Problems found</div>
        <div style={{ display: "grid", gap: 8 }}>{j.problems.map((p: any, i: number) => (
          <div className="problem" key={i}>
            <div className="p-head"><span className={`pill ${p.severity === "high" ? "bad" : p.severity === "medium" ? "warn" : "neutral"}`}>{p.severity}</span><span>{p.area}</span></div>
            <div>{p.description}</div>
            {p.evidence && <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>Evidence: {p.evidence}</div>}
            {p.fix && <div className="p-fix">Fix → {p.fix}</div>}
          </div>
        ))}</div></div>}
      {g?.points && <div className="card card-pad"><div className="card-title">Automated points</div>
        <div className="score-bars">{Object.entries(g.points).map(([k, v]: any) => {
          const max: Record<string, number> = { deliverables: 8, confirmation: 4, rebuild: 4, smoke: 10, task_checks: 4, env_py: 4, visual_check: 3, assets: 3 };
          return <div className="score-bar" key={k}><span>{k.replace("_", " ")}</span><div className="track"><div className="fill" style={{ width: `${(v / (max[k] || 10)) * 100}%` }} /></div><span className="mono">{v}</span></div>;
        })}</div></div>}
      {j?.strengths?.length > 0 && <div className="card card-pad"><div className="card-title">Strengths</div><ul style={{ margin: 0, paddingLeft: 18 }}>{j.strengths.map((s: string) => <li key={s}>{s}</li>)}</ul></div>}
    </div>
  );
}
