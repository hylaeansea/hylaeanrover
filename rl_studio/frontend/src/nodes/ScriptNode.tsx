import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useState } from "react";
import { useStudio } from "../context";
import type { GraphNode, NodeState } from "../types";
import ParamField from "./ParamField";

const ACCENTS: Record<string, string> = {
  training: "var(--blue)",
  eval: "var(--aqua)",
  promote: "var(--violet)",
};

export default function ScriptNode({
  id,
  data,
  selected,
}: NodeProps & { data: { gnode: GraphNode; state?: NodeState } }) {
  const { schema, actions } = useStudio();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [filter, setFilter] = useState("");
  const { gnode, state } = data;
  const spec = schema.node_types[gnode.type];
  if (!spec) return <div className="rf-node">unknown type {gnode.type}</div>;

  const params = gnode.params ?? {};
  const editable = spec.params.filter((p) => !p.is_input);
  const common = editable.filter((p) => p.common);
  const advanced = editable.filter((p) => !p.common);
  const shownAdvanced = filter
    ? advanced.filter(
        (p) =>
          p.name.includes(filter.toLowerCase()) ||
          p.help.toLowerCase().includes(filter.toLowerCase()),
      )
    : advanced;
  const overridden = advanced.filter((p) => params[p.name] !== undefined).length;
  const status = state?.status ?? "idle";
  const running = status === "running";

  return (
    <div
      className={`rf-node ${selected ? "selected" : ""}`}
      style={{ ["--accent" as string]: ACCENTS[gnode.type] ?? "var(--blue)" }}
    >
      {spec.accepts_input && <Handle type="target" position={Position.Left} />}
      <div className="head">
        <span className="title">{gnode.label ?? id}</span>
        <span className="kind">{spec.title}</span>
      </div>
      <div className="body nodrag nowheel">
        <div className={`status ${status}`}>
          <span className="dot" /> {status}
          {state?.returncode != null && status === "failed"
            ? ` (exit ${state.returncode})`
            : ""}
        </div>
        {common.map((p) => (
          <ParamField
            key={p.name}
            spec={p}
            value={params[p.name]}
            onChange={(v) => actions.updateParam(id, p.name, v)}
          />
        ))}
        {advanced.length > 0 && (
          <>
            <button
              className="advanced-toggle"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? "▾" : "▸"} advanced ({advanced.length}
              {overridden ? `, ${overridden} set` : ""})
            </button>
            {showAdvanced && (
              <div className="advanced">
                <input
                  className="search"
                  placeholder="filter params…"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                />
                {shownAdvanced.map((p) => (
                  <ParamField
                    key={p.name}
                    spec={p}
                    value={params[p.name]}
                    onChange={(v) => actions.updateParam(id, p.name, v)}
                  />
                ))}
              </div>
            )}
          </>
        )}
        {state?.last_metric?.ep_rew_mean != null && (
          <div className="metric">
            ep_rew_mean <b>{state.last_metric.ep_rew_mean.toFixed(1)}</b>
            {state.last_metric.step != null && (
              <span> @ {state.last_metric.step.toLocaleString()}</span>
            )}
          </div>
        )}
        <div className="node-actions">
          {running ? (
            <button className="danger" onClick={() => actions.stopNode(id)}>
              ■ Stop
            </button>
          ) : (
            <button className="primary" onClick={() => actions.runNode(id)}>
              ▶ Run
            </button>
          )}
          {state?.onnx && (
            <button onClick={() => actions.launchSim(state.onnx!)}>
              🎮 Sim
            </button>
          )}
          {state?.run_id && (
            <button className="small" onClick={() => actions.openRun(state.run_id!)}>
              logs
            </button>
          )}
        </div>
      </div>
      {spec.outputs.length > 0 && (
        <Handle type="source" position={Position.Right} />
      )}
    </div>
  );
}
