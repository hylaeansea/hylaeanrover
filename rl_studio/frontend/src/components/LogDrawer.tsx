import { useEffect, useRef, useState } from "react";
import { api, logsSocket } from "../api";
import type { Metric, RunSnapshot } from "../types";

const MAX_CLIENT_LINES = 5000;

function Sparkline({ metrics }: { metrics: Metric[] }) {
  if (metrics.length < 2) return null;
  const w = 216;
  const h = 44;
  const vals = metrics.map((m) => m.ep_rew_mean);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  const pts = vals
    .map(
      (v, i) =>
        `${((i / (vals.length - 1)) * (w - 4) + 2).toFixed(1)},${(
          h - 4 - ((v - min) / span) * (h - 8) + 2
        ).toFixed(1)}`,
    )
    .join(" ");
  const latest = vals[vals.length - 1];
  return (
    <div>
      <div className="sparkline-title">ep_rew_mean</div>
      <div className="sparkline-value">{latest.toFixed(1)}</div>
      <svg width={w} height={h} role="img" aria-label={`ep_rew_mean trend, latest ${latest.toFixed(1)}`}>
        <polyline
          points={pts}
          fill="none"
          stroke="var(--blue)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

function LogView({ runId }: { runId: string }) {
  const [lines, setLines] = useState<string[]>([]);
  const pre = useRef<HTMLPreElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    setLines([]);
    let ws: WebSocket | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let cursor = 0;
    let closed = false;

    const append = (newLines: string[]) => {
      if (!newLines.length) return;
      setLines((prev) => {
        const next = [...prev, ...newLines];
        return next.length > MAX_CLIENT_LINES
          ? next.slice(next.length - MAX_CLIENT_LINES)
          : next;
      });
    };

    const poll = async () => {
      try {
        const r = await api.logs(runId, cursor);
        cursor = r.next;
        append(r.lines);
        if (r.status !== "running" && pollTimer) clearInterval(pollTimer);
      } catch {
        if (pollTimer) clearInterval(pollTimer);
      }
    };

    try {
      ws = logsSocket(runId);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data) as { lines: string[] };
        append(msg.lines);
      };
      ws.onerror = () => {
        if (!closed && !pollTimer) pollTimer = setInterval(poll, 750);
      };
    } catch {
      pollTimer = setInterval(poll, 750);
    }
    return () => {
      closed = true;
      ws?.close();
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [runId]);

  useEffect(() => {
    const el = pre.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <pre
      className="log"
      ref={pre}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      }}
    >
      {lines.join("\n")}
    </pre>
  );
}

interface Props {
  runs: RunSnapshot[];
  openRun: RunSnapshot | null;
  onSelect: (id: string) => void;
  onClose: () => void;
}

export default function LogDrawer({ runs, openRun, onSelect, onClose }: Props) {
  const ordered = [...runs].sort((a, b) => b.started_at - a.started_at);
  return (
    <div className="drawer">
      <div className="tabs">
        {ordered.map((r) => (
          <span
            key={r.id}
            className={`tab ${openRun?.id === r.id ? "active" : ""}`}
            onClick={() => onSelect(r.id)}
          >
            {r.kind}
            {r.node_id ? `:${r.node_id}` : ""} · {r.status}
          </span>
        ))}
        <span style={{ flex: 1 }} />
        <button className="small" onClick={onClose}>✕</button>
      </div>
      {openRun ? (
        <div className="drawer-body">
          <LogView runId={openRun.id} />
          <div className="side">
            <div className={`status ${openRun.status}`}>
              <span className="dot" /> {openRun.status}
              {openRun.returncode != null ? ` (exit ${openRun.returncode})` : ""}
            </div>
            <Sparkline metrics={openRun.metrics} />
            {openRun.status === "running" && (
              <button
                className="danger"
                onClick={() => fetch(`/api/runs/${openRun.id}/stop`, { method: "POST" })}
              >
                ■ Stop run
              </button>
            )}
            <div className="argv">{openRun.argv.join(" ")}</div>
          </div>
        </div>
      ) : (
        <div className="empty-drawer">
          {runs.length ? "select a run tab" : "no runs yet — hit ▶ Run on a node"}
        </div>
      )}
    </div>
  );
}
