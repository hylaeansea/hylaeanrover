import type { FileNode, Graph, Schema, StateResponse } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  schema: () => fetch("/api/schema").then((r) => json<Schema>(r)),
  files: () => fetch("/api/files").then((r) => json<FileNode[]>(r)),
  graph: () => fetch("/api/graph").then((r) => json<Graph>(r)),
  saveGraph: (g: Graph) =>
    fetch("/api/graph", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(g),
    }).then((r) => json<{ ok: boolean }>(r)),
  state: () => fetch("/api/state").then((r) => json<StateResponse>(r)),
  runNode: (id: string) =>
    fetch(`/api/nodes/${id}/run`, { method: "POST" }).then((r) =>
      json<{ run_id: string; argv: string[] }>(r),
    ),
  stopNode: (id: string) =>
    fetch(`/api/nodes/${id}/stop`, { method: "POST" }).then((r) =>
      json<{ stopped: boolean }>(r),
    ),
  launchSim: (onnx: string) =>
    fetch("/api/sim", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ onnx }),
    }).then((r) => json<{ run_id: string }>(r)),
  logs: (runId: string, since: number) =>
    fetch(`/api/runs/${runId}/logs?since=${since}`).then((r) =>
      json<{ lines: string[]; next: number; status: string }>(r),
    ),
};

export function logsSocket(runId: string): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
}
