import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, fileUrl } from "../api";
import Markdown from "./Markdown";

const STAGES = [
  ["confirm", "Confirm"], ["assets", "Assets"], ["build", "Build"], ["smoke", "Smoke tests"], ["visual", "Visual check"], ["deliver", "Deliver"],
] as const;

function toolIcon(name: string) {
  if (name.startsWith("mcp__polypizza")) return "PP";
  if (name === "Bash") return "$";
  if (name === "Read") return "R";
  if (name === "Write") return "W";
  if (name === "Edit") return "E";
  if (name === "Skill") return "S";
  return "·";
}

const verdictCls: Record<string, string> = { found: "good", weak: "warn", not_found: "bad", error: "bad" };

function PolyPizza({ d, runId, images }: { d: any; runId: string; images?: string[] }) {
  return (
    <div className="pp-box">
      {d?.type === "find" && <div className="pp-roles">
        {Object.entries(d.roles || {}).map(([role, v]: any) => (
          <span key={role} className={`pill ${verdictCls[v.verdict] || "neutral"}`} title={(v.top || []).join(", ")}>{role}: {v.verdict?.replace("_", " ")}</span>
        ))}
      </div>}
      {d?.type === "search" && <div className="row" style={{ flexWrap: "wrap" }}>
        <span className={`pill ${verdictCls[d.verdict] || "neutral"}`}>“{d.query}” · {d.verdict?.replace("_", " ")}</span>
        <span className="muted" style={{ fontSize: 12 }}>{(d.top || []).join(" · ")}</span>
      </div>}
      {d?.type === "download" && <div className="row">
        {d.preview && <img src={d.preview} alt="" style={{ width: 96, margin: 0 }} />}
        <div style={{ fontSize: 12.5 }}>
          <div style={{ fontWeight: 600 }}>{d.title}</div>
          <div className="muted">{d.creator} · {d.licence}</div>
          {d.size && <div className="muted mono">{d.size.map((x: number) => x.toFixed(2)).join(" × ")} (file units)</div>}
        </div>
      </div>}
      {images?.map((im) => <img key={im} src={fileUrl(runId, im)} alt="Poly Pizza preview grid" />)}
    </div>
  );
}

export default function Timeline({ runId, run, onChange }: { runId: string; run: any; onChange: () => void }) {
  const [events, setEvents] = useState<any[]>([]);
  const [stage, setStage] = useState("confirm");
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<any>({});
  const [reply, setReply] = useState("");
  const [sending, setSending] = useState(false);
  const feed = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const next = useRef(0);

  useEffect(() => {
    setEvents([]); next.current = 0; stick.current = true;
    let alive = true;
    let timer: any;
    const tick = async () => {
      try {
        const r = await api.events(runId, next.current);
        if (!alive) return;
        if (r.events.length) setEvents((old) => [...old, ...r.events]);
        next.current = r.next;
        setStage(r.stage);
        setStatus(r.status || {});
        const wasRunning = running;
        setRunning(r.running);
        if (wasRunning && !r.running) onChange();
        const live = r.running || ["shot1", "shot2", "shot3", "running", "queued", "grading", "judging"].includes(r.status?.phase);
        timer = setTimeout(tick, live ? 1500 : 6000);
      } catch {
        timer = setTimeout(tick, 5000);
      }
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  useLayoutEffect(() => {
    const el = feed.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [events]);

  const onScroll = () => {
    const el = feed.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const results = new Map(events.filter((e) => e.kind === "tool_result").map((e) => [e.id, e]));
  const phase = status.phase || run?.phase;
  const isStudio = runId.startsWith("studio--");
  const canReply = isStudio && !running && ["awaiting_reply", "built", "error"].includes(phase);
  const stageIdx = phase === "done" || phase === "built" ? STAGES.length : STAGES.findIndex(([k]) => k === stage);

  const send = async (text: string) => {
    setSending(true);
    try { await api.reply(runId, text); setReply(""); setRunning(true); setStatus({ ...status, phase: "running" }); stick.current = true; onChange(); }
    catch (e: any) { alert(e.message); }
    finally { setSending(false); }
  };

  return (
    <>
      <div className="stages">
        {STAGES.map(([k, label], i) => (
          <div key={k} className={`stage ${i < stageIdx ? "done" : i === stageIdx ? "current" : ""}`}>
            <div className="bar" />{label}
          </div>
        ))}
      </div>
      <div className="feed" ref={feed} onScroll={onScroll}>
        {events.length === 0 && <div className="empty">{running ? "Starting a fresh agent…" : "No activity yet."}</div>}
        {events.map((e) => {
          if (e.kind === "shot_start") return <div key={e.i} className="msg user"><div className="who">You · message {e.shot}</div><Markdown text={e.prompt} /></div>;
          if (e.kind === "text") return <div key={e.i} className="msg agent"><div className="who">Agent</div><Markdown text={e.text} /></div>;
          if (e.kind === "shot_end") return <div key={e.i} className="shot-end">turn finished · {e.turns ?? "?"} steps{e.cost ? ` · $${Number(e.cost).toFixed(2)}` : ""}{e.timed_out ? " · timed out" : ""}</div>;
          if (e.kind === "tool") {
            const res = results.get(e.id);
            const pp = e.name.startsWith("mcp__polypizza");
            return (
              <div key={e.i}>
                <div className={`tool ${pp ? "pp" : ""} ${res && !res.ok ? "err" : ""}`} title={res?.text || ""}>
                  <div className="ti">{toolIcon(e.name)}</div>
                  <div style={{ minWidth: 0 }}><div className="tl">{e.label}</div>{e.detail && <div className="td">{e.detail}</div>}</div>
                </div>
                {pp && res?.polypizza && <PolyPizza d={res.polypizza} runId={runId} images={res.images} />}
                {pp && !res?.polypizza && res?.images && <PolyPizza d={null} runId={runId} images={res.images} />}
                {e.image_path && /(contact_sheet|filmstrip|collision_sheet)/.test(e.image_path) && <img className="inline" src={fileUrl(runId, e.image_path)} alt="" />}
              </div>
            );
          }
          return null;
        })}
      </div>
      {running && <div className="working"><span className="dot live" /> Agent is working{status.shot ? ` (message ${status.shot})` : ""}… the timeline updates live.</div>}
      {canReply && (
        <div className={`reply ${phase === "awaiting_reply" ? "ask" : ""}`}>
          <div style={{ fontSize: 12.5, marginBottom: 8, color: phase === "awaiting_reply" ? "var(--warn)" : "var(--muted)" }}>
            {phase === "awaiting_reply" ? "The agent is waiting for your answer." : "Ask for changes or follow-ups."}
          </div>
          <textarea value={reply} onChange={(e) => setReply(e.target.value)} placeholder={phase === "awaiting_reply" ? "Answer the questions, or just say go…" : "e.g. add two more trees and make the gates higher"}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && reply.trim()) send(reply); }} />
          <div className="row" style={{ marginTop: 8, justifyContent: "flex-end" }}>
            {phase === "awaiting_reply" && <button className="btn" disabled={sending} onClick={() => send("Looks good. Go ahead and build it with your recommended defaults.")}>Go with defaults</button>}
            <button className="btn primary" disabled={sending || !reply.trim()} onClick={() => send(reply)}>Send</button>
          </div>
        </div>
      )}
    </>
  );
}
