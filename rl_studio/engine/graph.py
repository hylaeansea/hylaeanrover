"""graph.json load/save/validate.

The graph is the committable artifact: nodes are serialized tool
invocations (params are sparse — only values that differ from the
schema defaults), edges are artifact transfer between nodes. The
engine knows nothing about what the params mean; the project
descriptor owns that.
"""

from __future__ import annotations

import json
import os
from typing import Any


class GraphError(ValueError):
    pass


def validate(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise GraphError("graph must have 'nodes' and 'edges' lists")
    ids: set[str] = set()
    for n in nodes:
        nid = n.get("id")
        if not isinstance(nid, str) or not nid:
            raise GraphError(f"node missing string id: {n!r}")
        if nid in ids:
            raise GraphError(f"duplicate node id {nid!r}")
        if not isinstance(n.get("type"), str):
            raise GraphError(f"node {nid!r} missing type")
        ids.add(nid)
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if src not in ids or dst not in ids:
            raise GraphError(f"edge references unknown node: {e!r}")
        if src == dst:
            raise GraphError(f"self-edge on {src!r}")
    return graph


def load(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"nodes": [], "edges": []}
    with open(path) as f:
        return validate(json.load(f))


def save(path: str, graph: dict[str, Any]) -> None:
    validate(graph)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(graph, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def parents_of(graph: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    by_id = {n["id"]: n for n in graph["nodes"]}
    return [by_id[e["from"]] for e in graph["edges"] if e["to"] == node_id]


def node_by_id(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    for n in graph["nodes"]:
        if n["id"] == node_id:
            return n
    raise GraphError(f"unknown node {node_id!r}")
