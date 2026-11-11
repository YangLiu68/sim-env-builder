export type RunSummary = {
  id: string;
  kind: "golden" | "studio";
  title: string;
  category?: string;
  prompt?: string;
  phase?: string;
  updated?: number;
  score?: number | null;
  verdict?: string | null;
  has_env: boolean;
  thumb?: string | null;
  iteration?: string | null;
};

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  runs: () => j<RunSummary[]>("/api/runs"),
  run: (id: string) => j<any>(`/api/runs/${id}`),
  events: (id: string, since = 0) => j<any>(`/api/runs/${id}/events?since=${since}`),
  golden: () => j<any>("/api/golden"),
  scene: (id: string) => j<any>(`/api/runs/${id}/scene`),
  newStudio: (prompt: string, category?: string) =>
    j<{ id: string }>("/api/studio", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, category }) }),
  reply: (id: string, text: string) =>
    j<any>(`/api/runs/${id}/reply`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }),
  evaluate: (id: string) => j<any>(`/api/runs/${id}/evaluate`, { method: "POST" }),
};

export const fileUrl = (id: string, path: string) => `/api/runs/${id}/file?path=${encodeURIComponent(path)}`;

export function timeAgo(t?: number | null) {
  if (!t) return "";
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export function scoreColor(v?: number | null) {
  if (v == null) return "var(--panel-2)";
  const t = Math.max(0, Math.min(1, (v - 40) / 55));
  const hue = 0 + t * 150;
  return `hsla(${hue}, 62%, 46%, 0.28)`;
}

export const phaseLabel: Record<string, string> = {
  queued: "Queued", running: "Agent working", shot1: "Confirming", shot2: "Building", shot3: "Finishing",
  awaiting_reply: "Waiting for you", built: "Built", grading: "Grading", judging: "Judging", done: "Done", error: "Error",
};

export function verdictPill(v?: string | null) {
  if (!v) return { cls: "neutral", label: "no smoke test" };
  if (v === "PASS") return { cls: "good", label: "Smoke PASS" };
  if (v === "PASS_WITH_WARNINGS") return { cls: "warn", label: "PASS w/ warnings" };
  return { cls: "bad", label: `Smoke ${v}` };
}
