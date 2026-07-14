import { useState } from "react";
import { useStudio } from "../context";
import type { FileNode } from "../types";

function BundleCard({ node }: { node: FileNode }) {
  const { actions } = useStudio();
  const tags = [
    node.has_onnx && "onnx",
    node.has_zip && "zip",
    node.has_vecnorm && "vecnorm",
  ].filter(Boolean);
  return (
    <div
      className="bundle-card"
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(
          "application/x-rlstudio-file",
          JSON.stringify({ path: node.path, rel: node.rel }),
        );
        e.dataTransfer.effectAllowed = "copy";
      }}
      title={`${node.rel}\ndrag onto the canvas to use as a training input`}
    >
      <span className="name">{node.rel}</span>
      <span className="tags">{tags.join(" · ")}</span>
      {node.has_onnx && (
        <button
          className="small"
          onClick={() => actions.launchSim(`${node.path}/model.onnx`)}
          title="launch the game with this policy"
        >
          🎮
        </button>
      )}
    </div>
  );
}

function collectBundles(node: FileNode, out: FileNode[]): FileNode[] {
  if (node.type === "dir") {
    if (node.bundle) out.push(node);
    node.children?.forEach((c) => collectBundles(c, out));
  }
  return out;
}

function Tree({ node, depth }: { node: FileNode; depth: number }) {
  const [open, setOpen] = useState(depth < 1);
  if (node.type === "file") {
    return <li className="file">{node.name}</li>;
  }
  return (
    <li>
      <span className="dirname" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} {node.name}
        {node.bundle ? " ●" : ""}
      </span>
      {open && node.children && (
        <ul>
          {node.children.map((c) => (
            <Tree key={c.path} node={c} depth={depth + 1} />
          ))}
        </ul>
      )}
    </li>
  );
}

export default function Sidebar({ files }: { files: FileNode[] }) {
  const bundles = files.flatMap((root) => collectBundles(root, []));
  return (
    <div className="sidebar">
      <h2>Saved models</h2>
      <div className="hint">
        Drag a bundle onto the canvas to warm-start a training/eval node from
        it, or 🎮 to play it in the game.
      </div>
      {bundles.length === 0 && <div className="hint">no bundles found yet</div>}
      {bundles.map((b) => (
        <BundleCard key={b.path} node={b} />
      ))}
      <h2>Files</h2>
      <ul className="ftree">
        {files.map((root) => (
          <Tree key={root.path} node={root} depth={0} />
        ))}
      </ul>
    </div>
  );
}
