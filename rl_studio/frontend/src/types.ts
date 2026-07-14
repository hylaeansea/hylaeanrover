export interface ParamSpec {
  name: string;
  flag: string;
  type: "int" | "float" | "str" | "flag";
  default: number | string | boolean | null;
  choices: string[] | null;
  required: boolean;
  help: string;
  common: boolean;
  is_input: boolean;
}

export interface NodeTypeSpec {
  title: string;
  script?: string;
  params: ParamSpec[];
  outputs: string[];
  accepts_input: boolean;
}

export interface Schema {
  project: string;
  describe: {
    ok: boolean;
    stages: string[];
    obs_dim: number | null;
    n_actions: number | null;
    error?: string;
    hint?: string;
  };
  node_types: Record<string, NodeTypeSpec>;
}

export interface GraphNode {
  id: string;
  type: string;
  label?: string;
  pos: [number, number];
  params?: Record<string, number | string | boolean>;
  path?: string;
}

export interface GraphEdge {
  from: string;
  to: string;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Metric {
  ep_rew_mean: number;
  step: number | null;
}

export interface NodeState {
  status: "idle" | "running" | "done" | "failed" | "stopped";
  run_id?: string;
  returncode?: number | null;
  last_metric?: Metric | null;
  onnx: string | null;
  has_bundle: boolean;
}

export interface RunSnapshot {
  id: string;
  node_id: string | null;
  kind: string;
  argv: string[];
  status: string;
  returncode: number | null;
  started_at: number;
  n_lines: number;
  metrics: Metric[];
}

export interface StateResponse {
  nodes: Record<string, NodeState>;
  runs: RunSnapshot[];
}

export interface FileNode {
  name: string;
  path: string;
  rel: string;
  type: "dir" | "file";
  size?: number | null;
  bundle?: boolean;
  has_onnx?: boolean;
  has_zip?: boolean;
  has_vecnorm?: boolean;
  children?: FileNode[];
}
