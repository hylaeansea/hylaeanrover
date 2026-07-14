import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStudio } from "../context";
import type { GraphNode, NodeState } from "../types";

export default function ArtifactNode({
  id,
  data,
  selected,
}: NodeProps & { data: { gnode: GraphNode; state?: NodeState } }) {
  const { actions } = useStudio();
  const { gnode, state } = data;
  return (
    <div
      className={`rf-node ${selected ? "selected" : ""}`}
      style={{ ["--accent" as string]: "var(--amber)" }}
    >
      <div className="head">
        <span className="title">{gnode.label ?? gnode.path ?? id}</span>
        <span className="kind">model</span>
      </div>
      <div className="body nodrag">
        <div className="artifact-files">{gnode.path}</div>
        <div className="node-actions">
          {state?.onnx && (
            <button onClick={() => actions.launchSim(state.onnx!)}>
              🎮 Sim
            </button>
          )}
          {state?.has_bundle ? (
            <span className="status done"><span className="dot" /> bundle</span>
          ) : (
            <span className="status"><span className="dot" /> no model.zip</span>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
