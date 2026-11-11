import { useEffect, useMemo, useState } from "react";
import { api, phaseLabel, scoreColor } from "../api";

function Trend({ iters }: { iters: any[] }) {
  const pts = iters.map((it, i) => ({ i, v: it.mean_total as number | null, label: it.iter.replace("iter_", "#") }));
  const W = 560, H = 170, P = 30;
  const xs = (i: number) => P + (pts.length <= 1 ? (W - 2 * P) / 2 : (i * (W - 2 * P)) / (pts.length - 1));
  const ys = (v: number) => H - P - (v / 100) * (H - 2 * P);
  const valid = pts.filter((p) => p.v != null);
  const path = valid.map((p, k) => `${k ? "L" : "M"}${xs(p.i)},${ys(p.v!)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 170 }}>
      {[0, 25, 50, 75, 100].map((g) => <g key={g}><line x1={P} x2={W - P} y1={ys(g)} y2={ys(g)} stroke="var(--line)" /><text x={6} y={ys(g) + 4} fontSize="10" fill="var(--muted)">{g}</text></g>)}
      <path d={path} fill="none" stroke="url(#gr)" strokeWidth="2.5" strokeLinejoin="round" />
      <defs><linearGradient id="gr" x1="0" x2="1"><stop offset="0" stopColor="#7aa2ff" /><stop offset="1" stopColor="#a58bff" /></linearGradient></defs>
      {valid.map((p) => <g key={p.i}><circle cx={xs(p.i)} cy={ys(p.v!)} r="4.5" fill="#0a0d13" stroke="#7aa2ff" strokeWidth="2" /><text x={xs(p.i)} y={ys(p.v!) - 10} fontSize="11" textAnchor="middle" fill="var(--text)">{p.v}</text></g>)}
      {pts.map((p) => <text key={p.i} x={xs(p.i)} y={H - 8} fontSize="10.5" textAnchor="middle" fill="var(--muted)">{p.label}</text>)}
    </svg>
  );
}

export default function Golden({ go }: { go: (x: string) => void }) {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    const load = () => api.golden().then(setData).catch(() => {});
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);
  const iters = data?.board?.iterations || [];
  const tasks = data?.tasks || [];
  const latest = [...iters].reverse().find((it: any) => it.n_done && !it.void) || iters[iters.length - 1];
  const best = Math.max(0, ...iters.map((it: any) => it.mean_total || 0));
  const problems = useMemo(() => {
    const out: any[] = [];
    (latest?.rows || []).forEach((r: any) => (r.problems || []).forEach((p: any) => out.push({ ...p, task: r.task })));
    const rank: Record<string, number> = { high: 0, medium: 1, low: 2 };
    return out.sort((a, b) => (rank[a.severity] ?? 3) - (rank[b.severity] ?? 3)).slice(0, 14);
  }, [latest]);
  if (!data) return <div className="page"><div className="skeleton" style={{ height: 380 }} /></div>;
  const cell = (task: string, it: any) => it.rows.find((r: any) => r.task === task);
  return (
    <div className="page">
      <div className="page-head">
        <div><h1 className="h1">Golden set</h1><div className="sub">{tasks.length} reference tasks across categories. Every iteration runs fresh agents with only the playbook, the Poly Pizza MCP and the task prompt; score = automated checks (40) + independent judge (60).</div></div>
      </div>
      <div className="kpis">
        <div className="card kpi"><div className="k-label">Iterations</div><div className="k-value">{iters.length}</div><div className="k-note">self-improvement rounds</div></div>
        <div className="card kpi"><div className="k-label">Latest mean score</div><div className="k-value">{latest?.mean_total ?? "–"}</div><div className="k-note">{latest?.iter} · {latest?.n_done}/{latest?.n} tasks scored</div></div>
        <div className="card kpi"><div className="k-label">Best mean score</div><div className="k-value">{best || "–"}</div><div className="k-note">target ≥ 85</div></div>
        <div className="card kpi"><div className="k-label">Categories covered</div><div className="k-value">{new Set(tasks.map((t: any) => t.category)).size}</div><div className="k-note">self-driving, arms, drones, legged…</div></div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 1fr)", gap: 14, marginTop: 14 }}>
        <div className="card card-pad"><div className="card-title">Mean score per iteration</div><Trend iters={iters} /></div>
        <div className="card card-pad" style={{ maxHeight: 250, overflow: "auto" }}>
          <div className="card-title">Iteration notes</div>
          {iters.map((it: any) => <div key={it.iter} style={{ fontSize: 13, padding: "4px 0", borderBottom: "1px solid var(--line)" }}><b>{it.iter}</b> <span className="muted">{it.notes}</span></div>)}
        </div>
      </div>
      <div className="card" style={{ marginTop: 14, overflow: "auto" }}>
        <table className="t matrix">
          <thead><tr><th>Task</th><th>Category</th><th>Tier</th>{iters.map((it: any) => <th key={it.iter} style={{ textAlign: "center" }}>{it.iter.replace("iter_", "#")}</th>)}</tr></thead>
          <tbody>
            {tasks.map((t: any) => (
              <tr key={t.id}>
                <td style={{ fontWeight: 500 }} title={t.prompt}>{t.id}</td>
                <td className="muted">{t.category}</td>
                <td><span className={`pill ${t.tier === "core" ? "info" : "neutral"}`}>{t.tier}</span></td>
                {iters.map((it: any) => {
                  const c = cell(t.id, it);
                  if (!c) return <td key={it.iter} />;
                  if (it.void) return <td key={it.iter} className="cell muted" title={it.notes} onClick={() => go(`run/golden--${it.iter}--${t.id}`)}
                    style={{ fontWeight: 400, fontSize: 12 }}>void</td>;
                  const pending = c.total == null;
                  return (
                    <td key={it.iter} className="cell" onClick={() => go(`run/golden--${it.iter}--${t.id}`)} title={c.summary || c.phase}
                      style={{ background: scoreColor(c.total) }}>
                      {pending ? <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>{phaseLabel[c.phase] || c.phase || "…"}</span> : c.total}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {problems.length > 0 && <div className="card card-pad" style={{ marginTop: 14 }}>
        <div className="card-title">What the judge flagged in {latest?.iter}</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(380px, 1fr))", gap: 8 }}>
          {problems.map((p, i) => (
            <div className="problem" key={i}>
              <div className="p-head"><span className={`pill ${p.severity === "high" ? "bad" : p.severity === "medium" ? "warn" : "neutral"}`}>{p.severity}</span><span>{p.task}</span><span>· {p.area}</span></div>
              <div style={{ fontSize: 13 }}>{p.description}</div>
              {p.fix && <div className="p-fix">Fix → {p.fix}</div>}
            </div>
          ))}
        </div>
      </div>}
    </div>
  );
}
