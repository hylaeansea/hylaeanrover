"""rl_studio server — run from repo root with python/.venv active:

    cd rl_studio && ../python/.venv/bin/python -m uvicorn server:app --port 8321

Serves the built frontend from frontend/dist when present; during
frontend development run `npm run dev` in frontend/ instead (Vite
proxies /api and /ws here).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import executor as executor_mod
from engine import files as files_mod
from engine import graph as graph_mod
from projects import hylaeanrover as project

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(HERE, "graph.json")

app = FastAPI(title="rl_studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = executor_mod.Executor()


# --- schema / files / graph ---------------------------------------------

@app.get("/api/schema")
def get_schema() -> dict[str, Any]:
    return {
        "project": "hylaeanrover",
        "describe": project.describe(),
        "node_types": project.node_types(),
    }


@app.get("/api/files")
def get_files() -> list[dict[str, Any]]:
    return files_mod.tree(project.FILE_ROOTS)


@app.get("/api/graph")
def get_graph() -> dict[str, Any]:
    return graph_mod.load(GRAPH_PATH)


@app.put("/api/graph")
async def put_graph(graph: dict[str, Any]) -> dict[str, Any]:
    try:
        graph_mod.save(GRAPH_PATH, graph)
    except graph_mod.GraphError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True}


# --- launching -----------------------------------------------------------

def _launch_node(node_id: str) -> executor_mod.Run:
    graph = graph_mod.load(GRAPH_PATH)
    node = graph_mod.node_by_id(graph, node_id)
    ntype = node["type"]
    if ntype not in project.SCRIPTS:
        raise HTTPException(status_code=422, detail=f"{ntype!r} nodes are not runnable")
    state = executor.node_states().get(node_id)
    if state and state["status"] == "running":
        raise HTTPException(status_code=409, detail=f"{node_id} already running")
    try:
        inputs = project.resolve_inputs(node, graph_mod.parents_of(graph, node_id))
        argv, cwd = project.build_argv(ntype, node.get("params", {}), inputs)
    except (ValueError, graph_mod.GraphError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return executor.spawn(
        argv, cwd, kind=ntype, node_id=node_id,
        line_parser=project.parse_metric_line if ntype == "training" else None,
    )


@app.post("/api/nodes/{node_id}/run")
def run_node(node_id: str) -> dict[str, Any]:
    run = _launch_node(node_id)
    return {"run_id": run.id, "argv": run.argv, "cwd": run.cwd}


@app.post("/api/nodes/{node_id}/stop")
def stop_node(node_id: str) -> dict[str, Any]:
    state = executor.node_states().get(node_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"no runs for {node_id}")
    return {"stopped": executor.stop(state["run_id"])}


class SimRequest(BaseModel):
    onnx: str


@app.post("/api/sim")
def launch_sim(req: SimRequest) -> dict[str, Any]:
    onnx = os.path.abspath(
        req.onnx if os.path.isabs(req.onnx)
        else os.path.join(project.REPO_ROOT, req.onnx)
    )
    if not onnx.startswith(project.REPO_ROOT + os.sep):
        raise HTTPException(status_code=422, detail="policy must live inside the repo")
    if not onnx.endswith(".onnx") or not os.path.exists(onnx):
        raise HTTPException(status_code=422, detail=f"not a .onnx file: {onnx}")
    argv, cwd = project.build_sim_argv(onnx)
    run = executor.spawn(argv, cwd, kind="sim", node_id=None)
    return {"run_id": run.id, "argv": run.argv}


# --- state / logs ---------------------------------------------------------

@app.get("/api/state")
def get_state() -> dict[str, Any]:
    graph = graph_mod.load(GRAPH_PATH)
    states = executor.node_states()
    nodes: dict[str, Any] = {}
    for node in graph["nodes"]:
        nid = node["id"]
        entry = dict(states.get(nid) or {"status": "idle"})
        entry["onnx"] = project.find_onnx(node)
        out_dir = project.output_dir(node)
        entry["has_bundle"] = bool(
            out_dir and os.path.exists(os.path.join(out_dir, "model.zip"))
        )
        nodes[nid] = entry
    return {"nodes": nodes, "runs": executor.list()}


@app.get("/api/runs/{run_id}/logs")
def get_logs(run_id: str, since: int = 0) -> dict[str, Any]:
    run = executor.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    lines, cursor, status = executor.lines_since(run_id, since)
    return {
        "lines": lines,
        "next": cursor,
        "status": status,
        "metrics": run.metrics[-500:],
    }


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    if executor.get(run_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return {"stopped": executor.stop(run_id)}


@app.websocket("/ws/runs/{run_id}")
async def ws_logs(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    if executor.get(run_id) is None:
        await ws.close(code=4404)
        return
    cursor = 0
    try:
        while True:
            lines, cursor2, status = executor.lines_since(run_id, cursor)
            if lines or status != "running":
                await ws.send_json(
                    {"lines": lines, "status": status, "next": cursor2}
                )
            cursor = cursor2
            if status != "running" and not lines:
                break
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    await ws.close()


# --- static frontend (after API routes so they take precedence) ----------

dist = os.path.join(HERE, "frontend", "dist")
if os.path.isdir(dist):
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
