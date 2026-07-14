import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  addEdge,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
} from "@xyflow/react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api } from "./api";
import { StudioContext } from "./context";
import LogDrawer from "./components/LogDrawer";
import Sidebar from "./components/Sidebar";
import ArtifactNode from "./nodes/ArtifactNode";
import ScriptNode from "./nodes/ScriptNode";
import type {
  FileNode,
  Graph,
  GraphNode,
  NodeState,
  RunSnapshot,
  Schema,
} from "./types";

const nodeTypes = { script: ScriptNode, artifact: ArtifactNode };

type RFNode = Node<{ gnode: GraphNode; state?: NodeState }>;

function toRF(g: Graph): { nodes: RFNode[]; edges: Edge[] } {
  return {
    nodes: g.nodes.map((gn) => ({
      id: gn.id,
      type: gn.type === "artifact" ? "artifact" : "script",
      position: { x: gn.pos[0], y: gn.pos[1] },
      data: { gnode: gn },
    })),
    edges: g.edges.map((e) => ({
      id: `${e.from}->${e.to}`,
      source: e.from,
      target: e.to,
    })),
  };
}

function toGraph(nodes: RFNode[], edges: Edge[]): Graph {
  return {
    nodes: nodes.map((n) => ({
      ...n.data.gnode,
      pos: [Math.round(n.position.x), Math.round(n.position.y)],
    })),
    edges: edges.map((e) => ({ from: e.source, to: e.target })),
  };
}

export default function App() {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [files, setFiles] = useState<FileNode[]>([]);
  const [runs, setRuns] = useState<RunSnapshot[]>([]);
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<RFNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const { screenToFlowPosition } = useReactFlow();
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const artifactSeq = useRef(1);
  const hadRunning = useRef(false);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => setError(String(e)));
    api.files().then(setFiles).catch(() => {});
    api
      .graph()
      .then((g) => {
        const { nodes: ns, edges: es } = toRF(g);
        setNodes(ns);
        setEdges(es);
      })
      .catch((e) => setError(String(e)));
  }, [setNodes, setEdges]);

  // Poll run/node state; refresh the file tree when a run finishes so
  // freshly written bundles show up.
  useEffect(() => {
    const tick = async () => {
      try {
        const st = await api.state();
        setRuns(st.runs);
        setNodes((ns) =>
          ns.map((n) => {
            const s = st.nodes[n.id];
            return s ? { ...n, data: { ...n.data, state: s } } : n;
          }),
        );
        const running = st.runs.some((r) => r.status === "running");
        if (hadRunning.current && !running) {
          api.files().then(setFiles).catch(() => {});
        }
        hadRunning.current = running;
      } catch {
        /* server briefly unreachable; keep last state */
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, [setNodes]);

  const save = useCallback(async () => {
    await api.saveGraph(toGraph(nodesRef.current, edgesRef.current));
    setDirty(false);
  }, []);

  const actions = useMemo(
    () => ({
      updateParam(nodeId: string, name: string, value: number | string | boolean | null) {
        setNodes((ns) =>
          ns.map((n) => {
            if (n.id !== nodeId) return n;
            const params = { ...(n.data.gnode.params ?? {}) };
            if (value === null || value === "") delete params[name];
            else params[name] = value;
            return { ...n, data: { ...n.data, gnode: { ...n.data.gnode, params } } };
          }),
        );
        setDirty(true);
      },
      async runNode(nodeId: string) {
        setError(null);
        try {
          await api.saveGraph(toGraph(nodesRef.current, edgesRef.current));
          setDirty(false);
          const { run_id } = await api.runNode(nodeId);
          setOpenRunId(run_id);
          setDrawerOpen(true);
        } catch (e) {
          setError(String(e));
        }
      },
      async stopNode(nodeId: string) {
        try {
          await api.stopNode(nodeId);
        } catch (e) {
          setError(String(e));
        }
      },
      async launchSim(onnx: string) {
        setError(null);
        try {
          const { run_id } = await api.launchSim(onnx);
          setOpenRunId(run_id);
          setDrawerOpen(true);
        } catch (e) {
          setError(String(e));
        }
      },
      openRun(runId: string) {
        setOpenRunId(runId);
        setDrawerOpen(true);
      },
    }),
    [setNodes],
  );

  const isValidConnection = useCallback(
    (conn: Connection | Edge) => {
      if (!schema || !conn.source || !conn.target) return false;
      const src = nodesRef.current.find((n) => n.id === conn.source);
      const dst = nodesRef.current.find((n) => n.id === conn.target);
      if (!src || !dst) return false;
      const srcSpec = schema.node_types[src.data.gnode.type];
      const dstSpec = schema.node_types[dst.data.gnode.type];
      if (!srcSpec?.outputs.length || !dstSpec?.accepts_input) return false;
      // one parent per node — the backend resolves parents[0]
      return !edgesRef.current.some(
        (e) => e.target === conn.target && e.source !== conn.source,
      );
    },
    [schema],
  );

  const onConnect = useCallback(
    (conn: Connection) => {
      setEdges((es) => addEdge(conn, es));
      setDirty(true);
    },
    [setEdges],
  );

  const onDrop = useCallback(
    (ev: React.DragEvent) => {
      const raw = ev.dataTransfer.getData("application/x-rlstudio-file");
      if (!raw) return;
      ev.preventDefault();
      const file = JSON.parse(raw) as { path: string; rel: string };
      const pos = screenToFlowPosition({ x: ev.clientX, y: ev.clientY });
      const id = `artifact_${Date.now()}_${artifactSeq.current++}`;
      const gnode: GraphNode = {
        id,
        type: "artifact",
        label: file.rel,
        pos: [Math.round(pos.x), Math.round(pos.y)],
        path: file.path,
      };
      setNodes((ns) => [
        ...ns,
        { id, type: "artifact", position: pos, data: { gnode } },
      ]);
      setDirty(true);
    },
    [screenToFlowPosition, setNodes],
  );

  const addNode = useCallback(
    (type: string) => {
      if (!schema) return;
      const spec = schema.node_types[type];
      const id = `${type}_${Date.now() % 100000}`;
      const pos = screenToFlowPosition({
        x: window.innerWidth / 2,
        y: window.innerHeight / 3,
      });
      const params: Record<string, string | number | boolean> = {};
      for (const p of spec.params) {
        if (p.required && p.choices?.length) params[p.name] = p.choices[0];
      }
      const gnode: GraphNode = {
        id,
        type,
        label: `${spec.title} (new)`,
        pos: [Math.round(pos.x), Math.round(pos.y)],
        params,
      };
      setNodes((ns) => [
        ...ns,
        { id, type: "script", position: pos, data: { gnode } },
      ]);
      setDirty(true);
    },
    [schema, screenToFlowPosition, setNodes],
  );

  const openRun = runs.find((r) => r.id === openRunId) ?? null;

  if (!schema) {
    return (
      <div className="app">
        <div className="topbar"><h1>rl_studio</h1></div>
        <div style={{ padding: 24, color: "var(--text-2)" }}>
          {error ? `Failed to reach backend: ${error}` : "Loading…"}
        </div>
      </div>
    );
  }

  return (
    <StudioContext.Provider value={{ schema, actions }}>
      <div className="app">
        <div className="topbar">
          <h1>rl_studio</h1>
          <span className="env">{schema.project}</span>
          {schema.describe.ok ? (
            <span className="env">
              obs {schema.describe.obs_dim} · {schema.describe.n_actions} actions ·{" "}
              {schema.describe.stages.length} stages
            </span>
          ) : (
            <span className="rebuild-warning" title={schema.describe.error}>
              ⚠ native module unavailable — {schema.describe.hint}
            </span>
          )}
          <div className="spacer" />
          {error && (
            <span className="rebuild-warning" title={error}>⚠ {error}</span>
          )}
          <button onClick={() => addNode("training")}>+ Train</button>
          <button onClick={() => addNode("eval")}>+ Eval</button>
          <button onClick={() => addNode("promote")}>+ Promote</button>
          <button onClick={() => setDrawerOpen((v) => !v)}>
            {drawerOpen ? "Hide logs" : "Logs"}
          </button>
          <button className="primary" onClick={save} disabled={!dirty}>
            {dirty ? "Save graph" : "Saved"}
          </button>
        </div>
        <div className="main">
          <Sidebar files={files} />
          <div className="canvas-wrap" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={(ch) => {
                onNodesChange(ch);
                if (ch.some((c) => c.type !== "select" && c.type !== "dimensions"))
                  setDirty(true);
              }}
              onEdgesChange={(ch) => {
                onEdgesChange(ch);
                if (ch.some((c) => c.type === "remove")) setDirty(true);
              }}
              onConnect={onConnect}
              isValidConnection={isValidConnection}
              fitView
              minZoom={0.2}
              deleteKeyCode={["Backspace", "Delete"]}
              colorMode="dark"
              proOptions={{ hideAttribution: true }}
            >
              <Background gap={24} />
              <Controls />
              <MiniMap pannable zoomable />
            </ReactFlow>
          </div>
        </div>
        {drawerOpen && (
          <LogDrawer
            runs={runs}
            openRun={openRun}
            onSelect={setOpenRunId}
            onClose={() => setDrawerOpen(false)}
          />
        )}
      </div>
    </StudioContext.Provider>
  );
}
