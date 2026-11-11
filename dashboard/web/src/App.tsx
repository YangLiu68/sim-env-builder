import { useCallback, useEffect, useState } from "react";
import { api, phaseLabel, RunSummary } from "./api";
import Golden from "./pages/Golden";
import RunPage from "./pages/Run";
import Studio, { Card } from "./pages/Studio";

function useRoute(): [string, (r: string) => void] {
  const get = () => location.hash.replace(/^#\/?/, "") || "studio";
  const [route, setRoute] = useState(get);
  useEffect(() => {
    const on = () => setRoute(get());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [route, (r: string) => { location.hash = `/${r}`; }];
}

const Icon = {
  studio: <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 12.5l4-9 4 9M3.5 9.5h5" /><path d="M11 6.5l3 6" /></svg>,
  envs: <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="2" y="2" width="5" height="5" rx="1.2" /><rect x="9" y="2" width="5" height="5" rx="1.2" /><rect x="2" y="9" width="5" height="5" rx="1.2" /><rect x="9" y="9" width="5" height="5" rx="1.2" /></svg>,
  golden: <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 13.5h12M4 11V7M8 11V3.5M12 11V5.5" /></svg>,
};

function Envs({ runs, go }: { runs: RunSummary[]; go: (r: string) => void }) {
  const envs = runs.filter((r) => r.has_env);
  const [q, setQ] = useState("");
  const shown = envs.filter((r) => !q || `${r.title} ${r.category} ${r.prompt}`.toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="page">
      <div className="page-head">
        <div><h1 className="h1">Environments</h1><div className="sub">Every scene built so far, from the Studio and the golden set. Open one to drive it in the 3D viewer.</div></div>
        <input placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 240 }} />
      </div>
      {shown.length === 0 ? <div className="card empty">No environments yet.</div> : <div className="grid-cards">{shown.map((r) => <Card key={r.id} r={r} go={go} />)}</div>}
    </div>
  );
}

export default function App() {
  const [route, go] = useRoute();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const refresh = useCallback(() => { api.runs().then(setRuns).catch(() => {}); }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 10000); return () => clearInterval(t); }, [refresh]);
  const section = route.startsWith("run/") ? (route.includes("golden--") ? "golden" : "studio") : route;
  const recent = runs.slice(0, 18);
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><svg width="18" height="18" viewBox="0 0 32 32"><path d="M8 22l8-13 8 13z" fill="none" stroke="#7aa2ff" strokeWidth="2.6" strokeLinejoin="round" /><circle cx="16" cy="18.5" r="2.6" fill="#53d6a4" /></svg></div>
          <div><div className="brand-name">SimForge Studio</div><div className="brand-sub">sim-env-builder skill</div></div>
        </div>
        <nav className="nav">
          <button className={section === "studio" ? "active" : ""} onClick={() => go("studio")}>{Icon.studio} Studio</button>
          <button className={section === "envs" ? "active" : ""} onClick={() => go("envs")}>{Icon.envs} Environments</button>
          <button className={section === "golden" ? "active" : ""} onClick={() => go("golden")}>{Icon.golden} Golden set</button>
        </nav>
        <div className="side-title">Recent runs</div>
        <div className="recent">
          {recent.map((r) => {
            const live = ["shot1", "shot2", "shot3", "running", "queued", "grading", "judging"].includes(r.phase || "");
            const dot = live ? "live" : r.phase === "awaiting_reply" ? "warn" : r.verdict === "FAIL" || r.phase === "error" ? "bad" : r.has_env ? "good" : "";
            return (
              <button key={r.id} onClick={() => go(`run/${r.id}`)} title={r.prompt}>
                <span className={`dot ${dot}`} />
                <span style={{ minWidth: 0 }}>
                  <div className="r-title">{r.title}</div>
                  <div className="r-sub">{r.kind === "golden" ? r.iteration : "studio"} · {phaseLabel[r.phase || ""] || r.phase || "–"}{r.score != null ? ` · ${r.score}` : ""}</div>
                </span>
              </button>
            );
          })}
        </div>
      </aside>
      <main className="main">
        {route.startsWith("run/") ? <RunPage runId={route.slice(4)} go={go} key={route} />
          : section === "golden" ? <Golden go={go} />
          : section === "envs" ? <Envs runs={runs} go={go} />
          : <Studio runs={runs} go={go} refresh={refresh} />}
      </main>
    </div>
  );
}
