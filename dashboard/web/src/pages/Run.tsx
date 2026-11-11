import { useCallback, useEffect, useState } from "react";
import { api, phaseLabel, verdictPill } from "../api";
import { AssetsPanel, DocsPanel, JudgePanel, PhysicsPanel, ResultsPanel } from "../components/Panels";
import Timeline from "../components/Timeline";
import Viewer from "../components/Viewer";

const TABS = [["viewer", "3D viewer"], ["results", "Results"], ["physics", "Physics"], ["assets", "Assets"], ["docs", "Docs"], ["judge", "Evaluation"]] as const;

export default function RunPage({ runId, go }: { runId: string; go: (r: string) => void }) {
  const [run, setRun] = useState<any>(null);
  const [tab, setTab] = useState<string>("viewer");
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => api.run(runId).then((r) => { setRun(r); setErr(null); }).catch((e) => setErr(e.message)), [runId]);
  useEffect(() => { setRun(null); load(); }, [load]);
  useEffect(() => {
    const live = run && (run.running || ["shot1", "shot2", "shot3", "running", "queued", "grading", "judging"].includes(run.phase));
    const t = setInterval(load, live ? 4000 : 15000);
    return () => clearInterval(t);
  }, [run, load]);
  useEffect(() => { if (run && !run.has_env && tab === "viewer") setTab("results"); }, [run, tab]);

  if (err) return <div className="page"><div className="empty">Could not load run: {err}</div></div>;
  if (!run) return <div className="page"><div className="skeleton" style={{ height: 300 }} /></div>;
  const vp = verdictPill(run.verdict);
  const cost = run.status?.cost ?? run.run_meta?.agent_cost_usd;
  const live = run.running || ["shot1", "shot2", "shot3", "running", "queued"].includes(run.phase);

  return (
    <div className="run">
      <div className="run-left">
        <div className="run-head">
          <div className="row" style={{ marginBottom: 6 }}>
            <button className="btn ghost sm" onClick={() => go(run.kind === "golden" ? "golden" : "studio")}>← {run.kind === "golden" ? "Golden set" : "Studio"}</button>
            <span className="grow" />
            <span className={`pill ${live ? "info" : run.phase === "awaiting_reply" ? "warn" : run.phase === "error" ? "bad" : "neutral"}`}>
              {live && <span className="dot live" />}{phaseLabel[run.phase] || run.phase || "–"}
            </span>
          </div>
          <div className="t">{run.kind === "golden" ? run.task?.id : run.title}</div>
          <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>
            {[run.category, run.iteration, cost ? `$${Number(cost).toFixed(2)} agent cost` : null,
              run.run_meta?.shots_used ? `${run.run_meta.shots_used} messages` : null].filter(Boolean).join(" · ")}
          </div>
        </div>
        <Timeline runId={runId} run={run} onChange={load} />
      </div>
      <div className="run-right">
        <div className="row" style={{ padding: "8px 14px 0", gap: 8 }}>
          <div className="tabs grow" style={{ borderBottom: 0, padding: 0 }}>
            {TABS.map(([k, label]) => (
              <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)} disabled={k === "viewer" && !run.has_env}
                style={k === "viewer" && !run.has_env ? { opacity: 0.4 } : undefined}>{label}</button>
            ))}
          </div>
          <span className={`pill ${vp.cls}`}>{vp.label}</span>
          {run.score != null && <span className="pill info">score {run.score}</span>}
        </div>
        <div style={{ borderBottom: "1px solid var(--line)" }} />
        <div className="tab-body" style={tab === "viewer" ? { overflow: "hidden" } : undefined}>
          {tab === "viewer" && run.has_env && <Viewer runId={runId} key={`${runId}-${run.env?.manifest?.geoms}`} />}
          {tab === "results" && <ResultsPanel runId={runId} run={run} />}
          {tab === "physics" && <PhysicsPanel run={run} />}
          {tab === "assets" && <AssetsPanel run={run} />}
          {tab === "docs" && <DocsPanel run={run} />}
          {tab === "judge" && <JudgePanel run={run} onEvaluate={run.kind === "studio" && run.has_env ? () => api.evaluate(runId).then(load) : undefined} />}
        </div>
      </div>
    </div>
  );
}
