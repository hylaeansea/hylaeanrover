# rl_studio

Local-first web UI for the RL training curriculum ([issue #18](https://github.com/hylaeansea/hylaeanrover/issues/18)).
The training pipeline in `python/TRAINING_GUIDE.md` is a DAG — train →
evaluate → promote per stage, with each stage warm-started from the
previous stage's promoted bundle. rl_studio renders that DAG as an
editable node graph: click ▶ Run on a node instead of assembling an
80-flag `train.py` command line, watch live logs and `ep_rew_mean`,
and launch the game with any saved model via its 🎮 button.

![rl_studio](../docs/rl_studio.png)

## Prerequisites

- `python/.venv` set up with `maturin develop --release` run (see
  `python/README.md`). The server imports `hylaeanrover` in-process; if
  the native module is stale the top bar shows a rebuild warning.
- `fastapi` + `uvicorn` in that venv:
  `uv pip install --python python/.venv/bin/python fastapi 'uvicorn[standard]'`
- Node ≥ 20 for the frontend (only needed to build/develop it).

## Run it

```bash
# one-time (or after frontend changes): build the UI
cd rl_studio/frontend && npm install && npm run build

# start the server (serves the built UI at http://localhost:8321)
cd rl_studio && ../python/.venv/bin/python -m uvicorn server:app --port 8321
```

For frontend development, run `npm run dev` in `frontend/` instead and
open the Vite URL — `/api` and `/ws` are proxied to :8321.

## What you can do

- **Run curriculum stages** — each training node is a `train.py`
  invocation. Common params are inline; the other ~70 flags live under
  "advanced" with a filter box. Params left blank use the script's
  defaults; only overrides are stored in `graph.json`.
- **Wire nodes** — an edge feeds the parent's output bundle into the
  child (`--load`/`--vecnorm` for train/eval, `--run` for promote).
  One parent per node.
- **Evaluate / promote** — eval nodes run `evaluate.py`; promote nodes
  run `promote_model.py`, producing the game-launchable
  `models/<stage>/model.onnx`.
- **Launch the game with a saved model** — 🎮 on any node or sidebar
  bundle runs `cargo run -p hylaeanrover_game --release -- --policy <onnx>`
  (first launch may compile).
- **Drag stored models onto the canvas** — sidebar bundles become
  artifact nodes; connect one to a training node to warm-start from it.
- **Watch runs** — the log drawer streams stdout over WebSocket with an
  `ep_rew_mean` sparkline; nodes show live status and last metric.

## Layout

```
rl_studio/
├── engine/              # generic: graph file, subprocess registry, file tree
├── projects/
│   └── hylaeanrover.py  # the only project-aware code; introspects the real
│                        #   argparse parsers (build_parser()) so the UI never
│                        #   drifts from the scripts
├── server.py            # FastAPI app + WebSocket log streaming
├── graph.json           # the curriculum DAG (committed, mirrors TRAINING_GUIDE)
└── frontend/            # Vite + React + @xyflow/react
```

The engine/frontend are project-agnostic (schema-driven); supporting
another project means adding a descriptor module, not refactoring.
