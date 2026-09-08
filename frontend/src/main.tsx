import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowUpRight,
  Bot,
  Check,
  ChevronRight,
  CircleAlert,
  Cloud,
  Cpu,
  Gauge,
  GitBranch,
  Radio,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Terminal,
  X,
} from "lucide-react";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
const phases = [
  { id: "benchmark", label: "Benchmark", short: "01", agent: "Atlas", color: "cyan" },
  { id: "analyze_failures", label: "Failure map", short: "02", agent: "Rook", color: "violet" },
  { id: "research", label: "Research", short: "03", agent: "Mica", color: "lime" },
  { id: "curate_data", label: "Data curation", short: "04", agent: "Vale", color: "amber" },
  { id: "design_training", label: "Training design", short: "05", agent: "Kite", color: "pink" },
  { id: "execute_training", label: "SageMaker", short: "06", agent: "Forge", color: "orange" },
  { id: "evaluate", label: "Held-out eval", short: "07", agent: "Lens", color: "blue" },
  { id: "promote_decision", label: "Promotion", short: "08", agent: "Crown", color: "green" },
] as const;

type PhaseId = (typeof phases)[number]["id"];
type RunStatus = "idle" | "created" | "running" | "completed" | "failed" | "blocked" | "cancelled";
type Run = {
  runId: string;
  targetModel?: string;
  environment?: string;
  objective?: string;
  currentPhase?: string;
  status: RunStatus;
  createdAt?: string;
  updatedAt?: string;
  baselinePerformance?: number | null;
  championPerformance?: number | null;
  candidatePerformance?: number | null;
  totalCostUSD?: number;
  totalTrainingTimeMin?: number;
  championCheckpoint?: string | null;
};
type CompareRun = {
  run_id: string;
  run_number: number;
  baseline_aggregate?: number | null;
  candidate_aggregate?: number | null;
  decision?: string | null;
  status?: string;
};
type ComparePayload = { rows?: CompareRun[]; total?: number };
type Health = { status?: string; aws_region?: string; components?: Record<string, string> };

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "content-type": "application/json", ...(options?.headers ?? {}) },
  });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json() as Promise<T>;
}

function phaseIndex(current?: string): number {
  const index = phases.findIndex((phase) => phase.id === current);
  return index < 0 ? -1 : index;
}

function formatMetric(value?: number | null): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function formatTime(value?: number): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}m` : "—";
}

function App() {
  const [run, setRun] = useState<Run | null>(null);
  const [comparison, setComparison] = useState<ComparePayload | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [model, setModel] = useState("google/functiongemma-270m-it");
  const [checkpoint, setCheckpoint] = useState("s3://post-training/checkpoints/functiongemma-base");
  const [environment, setEnvironment] = useState("agentgym-service-recovery");
  const [events, setEvents] = useState<string[]>([]);
  const eventSource = useRef<EventSource | null>(null);

  const loadComparison = useCallback(async () => {
    try {
      setComparison(await api<ComparePayload>("/api/runs/compare"));
    } catch {
      // An empty run registry is an honest empty state, not a fabricated chart.
      setComparison(null);
    }
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api<Health>("/health"));
    } catch {
      setHealth(null);
    }
  }, []);

  const loadRun = useCallback(async (runId: string) => {
    try {
      const next = await api<Run>(`/api/runs/${encodeURIComponent(runId)}`);
      setRun(next);
      setEvents((current) => [`${new Date().toLocaleTimeString()}  ${next.currentPhase ?? "idle"}  ${next.status}`, ...current].slice(0, 8));
      return next;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not read run");
      return null;
    }
  }, []);

  useEffect(() => {
    void loadHealth();
    void loadComparison();
    const interval = window.setInterval(() => {
      void loadHealth();
      if (run?.runId) void loadRun(run.runId);
      void loadComparison();
    }, 3000);
    return () => window.clearInterval(interval);
  }, [loadComparison, loadHealth, loadRun, run?.runId]);

  useEffect(() => () => eventSource.current?.close(), []);

  const startRun = async () => {
    setConnecting(true);
    setError(null);
    try {
      const created = await api<{ runId: string }>(
        `/api/runs?target_model=${encodeURIComponent(model)}&base_checkpoint=${encodeURIComponent(checkpoint)}&environment=${encodeURIComponent(environment)}`,
        { method: "POST" },
      );
      const first = await loadRun(created.runId);
      if (first) subscribeToRun(created.runId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Run could not be created");
    } finally {
      setConnecting(false);
    }
  };

  const subscribeToRun = (runId: string) => {
    eventSource.current?.close();
    // The AWS coordinator may expose SSE in a deployed environment. Local mode
    // intentionally falls back to the same safe status polling path.
    const stream = new EventSource(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/events`);
    eventSource.current = stream;
    stream.onmessage = (message) => {
      if (message.data) setEvents((current) => [`${new Date().toLocaleTimeString()}  event received`, ...current].slice(0, 8));
      void loadRun(runId);
    };
    stream.onerror = () => stream.close();
  };

  const stepRun = async () => {
    if (!run?.runId) return;
    setError(null);
    try {
      await api(`/api/runs/${encodeURIComponent(run.runId)}/step`, { method: "POST" });
      await loadRun(run.runId);
      await loadComparison();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Phase could not advance");
    }
  };

  const cancelRun = async () => {
    if (!run?.runId) return;
    try {
      await api(`/api/runs/${encodeURIComponent(run.runId)}/cancel`, { method: "POST" });
      await loadRun(run.runId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Run could not be cancelled");
    }
  };

  const activeIndex = phaseIndex(run?.currentPhase);
  const runIsActive = run?.status === "running" || run?.status === "created";
  const chartRows = comparison?.rows ?? [];
  const champion = run?.championPerformance ?? run?.baselinePerformance;

  return (
    <main className="console-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><Sparkles size={18} /></div>
          <div><p className="eyebrow">Autonomous post-training</p><h1>Control room</h1></div>
        </div>
        <div className="topbar-meta"><span className={`connection-dot ${health ? "online" : ""}`} /> <span>{health ? "AWS control plane connected" : "Awaiting control plane"}</span><span className="divider" /><span>Nemotron Super 3 · 120B</span></div>
      </header>

      <section className="hero-grid">
        <div className="hero-copy">
          <p className="eyebrow">Live execution bench / 5-run budget</p>
          <h2>Watch the model<br /><em>get better.</em></h2>
          <p className="hero-description">Eight specialist agents turn observed failures into a measured candidate. Every handoff is visible. Every promotion is earned.</p>
          <div className="hero-actions"><button className="primary-button" onClick={startRun} disabled={connecting || Boolean(runIsActive)}><Radio size={16} /> {connecting ? "Launching…" : runIsActive ? "Run in progress" : "Launch a run"}</button>{runIsActive && <button className="ghost-button" onClick={cancelRun}><X size={16} /> Stop safely</button>}</div>
          {error && <div className="error-strip"><CircleAlert size={16} /> {error}<button onClick={() => setError(null)} aria-label="Dismiss error"><X size={14} /></button></div>}
        </div>
        <div className="launch-card">
          <div className="card-header"><span>Run launcher</span><span className="status-chip">guarded</span></div>
          <label>Target model<input value={model} onChange={(event) => setModel(event.target.value)} /></label>
          <label>Base checkpoint<input value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)} /></label>
          <label>Environment<select value={environment} onChange={(event) => setEnvironment(event.target.value)}><option value="agentgym-service-recovery">AgentGym / service recovery</option><option value="agent-eval-v1">AgentEval v1</option><option value="webshop">WebShop</option></select></label>
          <p className="field-note"><ShieldCheck size={14} /> Checkpoint provenance is verified before compute starts.</p>
        </div>
      </section>

      <section className="bench-section">
        <div className="section-heading"><div><p className="eyebrow">The bench</p><h3>Agents at work</h3></div><div className="live-indicator"><span className="pulse" /> {runIsActive ? "executing" : "standby"}</div></div>
        <div className="stage-rail" aria-label="Post-training execution phases">
          <div className="rail-line" style={{ "--progress": `${Math.max(0, activeIndex) / (phases.length - 1) * 100}%` } as CSSProperties} />
          {phases.map((phase, index) => {
            const completed = activeIndex > index || run?.status === "completed";
            const active = activeIndex === index && runIsActive;
            return <div className={`stage-node ${completed ? "completed" : ""} ${active ? "active" : ""}`} key={phase.id}>
              <div className={`bot bot-${phase.color}`}><Bot size={18} /><span className="bot-spark" /></div>
              <div className="stage-copy"><span className="stage-number">{phase.short}</span><strong>{phase.label}</strong><small>{phase.agent}</small></div>
              {completed && <Check className="stage-check" size={14} />}
            </div>;
          })}
        </div>
        <div className="bench-footer"><div><span className="muted-label">Current handoff</span><strong>{run?.currentPhase?.replaceAll("_", " ") ?? "No run selected"}</strong></div><div className="bench-note"><Activity size={15} /> live changes appear in run telemetry</div><button className="text-button" onClick={stepRun} disabled={!runIsActive}>Advance phase <ChevronRight size={15} /></button></div>
      </section>

      <section className="data-grid">
        <div className="panel telemetry-panel"><div className="panel-heading"><div><p className="eyebrow">Observation stream</p><h3>Run telemetry</h3></div><Terminal size={17} /></div><div className="telemetry-list">{events.length ? events.map((event, index) => <div className="telemetry-row" key={`${event}-${index}`}><span className="telemetry-led" /><code>{event}</code></div>) : <div className="empty-panel"><Activity size={20} /><p>Telemetry will appear when an agent starts moving.</p></div>}</div><div className="panel-footer"><span>Metadata only · prompts and task contents stay sealed</span><button className="icon-button" onClick={() => run?.runId && void loadRun(run.runId)} aria-label="Refresh telemetry"><RefreshCw size={15} /></button></div></div>
        <div className="panel score-panel"><div className="panel-heading"><div><p className="eyebrow">Objective signal</p><h3>Champion vs candidate</h3></div><Gauge size={17} /></div><div className="score-hero"><span className="score-value">{formatMetric(champion)}</span><span className="score-label">current champion</span></div><div className="score-bars"><div className="score-bar-row"><span>baseline</span><div className="bar-track"><div className="bar-fill baseline" style={{ width: `${Math.max(0, Math.min(100, (run?.baselinePerformance ?? 0) * 100))}%` }} /></div><strong>{formatMetric(run?.baselinePerformance)}</strong></div><div className="score-bar-row"><span>candidate</span><div className="bar-track"><div className="bar-fill candidate" style={{ width: `${Math.max(0, Math.min(100, (run?.candidatePerformance ?? 0) * 100))}%` }} /></div><strong>{formatMetric(run?.candidatePerformance)}</strong></div></div><div className="score-meta"><span><Cloud size={14} /> {formatTime(run?.totalTrainingTimeMin)} compute</span><span><ShieldCheck size={14} /> deterministic gate</span></div></div>
        <div className="panel comparison-panel"><div className="panel-heading"><div><p className="eyebrow">History / max 5</p><h3>Run comparison</h3></div><GitBranch size={17} /></div>{chartRows.length ? <div className="comparison-list">{chartRows.map((row) => <div className="comparison-row" key={row.run_id}><span className="run-tag">R{row.run_number}</span><div className="comparison-bars"><div className="mini-track"><i className="mini-fill baseline" style={{ width: `${Math.max(0, Math.min(100, (row.baseline_aggregate ?? 0) * 100))}%` }} /></div><div className="mini-track"><i className="mini-fill candidate" style={{ width: `${Math.max(0, Math.min(100, (row.candidate_aggregate ?? 0) * 100))}%` }} /></div></div><span className={`decision ${row.decision === "PROMOTE" ? "promote" : ""}`}>{row.decision ?? row.status ?? "pending"}</span></div>)}</div> : <div className="empty-panel graph-empty"><GitBranch size={24} /><p>Complete a run to draw the comparison graph.</p><span>No metrics are invented for the demo.</span></div>}<div className="legend"><span><i className="legend-dot baseline" /> baseline</span><span><i className="legend-dot candidate" /> candidate</span></div></div>
      </section>

      <footer className="footer-bar"><span><Cpu size={14} /> AWS / {health?.aws_region ?? "us-east-1"}</span><span>Target: FunctionGemma</span><span>Evidence: {run?.status === "completed" ? "LIVE" : "awaiting live run"}</span><a href="https://github.com" target="_blank" rel="noreferrer">View runbook <ArrowUpRight size={13} /></a></footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
