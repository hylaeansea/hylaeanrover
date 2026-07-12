"""Promote a training run's result into the tracked `models/<stage>/`
bundle — the canonical "best so far" for a stage.

The bundle holds everything needed to resume training, run the in-game
autopilot, and share with other developers:

    models/<stage>/model.zip        SB3 policy (resume training / export)
    models/<stage>/vecnorm.pkl      matching VecNormalize obs stats
    models/<stage>/model.onnx       policy for the in-game autopilot
    models/<stage>/model.norm.json  obs stats + frame-skip for the autopilot

These are small (~240 KB/stage) and meant to be committed to git.

Examples
--------
Promote the eval-best checkpoint of a run (recommended)::

    python examples/promote_model.py --stage locomotion --run runs/stage0

Promote the final model instead, recording the training frame-skip::

    python examples/promote_model.py --stage locomotion --run runs/stage0 \
        --source final --frame-skip 4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hylaeanrover.export import export_policy
from hylaeanrover.wrappers import STAGES

# Repo root is two levels up from python/examples/.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_pair(run: Path, source: str) -> tuple[Path, Path]:
    """Return (model.zip, vecnorm.pkl) for the chosen source."""
    if source == "best":
        model, vecnorm = run / "best" / "best_model.zip", run / "best" / "vecnorm.pkl"
    else:
        model, vecnorm = run / "model.zip", run / "vecnorm.pkl"
    if not model.exists():
        raise SystemExit(f"{model} not found (did the run reach an eval / finish?)")
    if not vecnorm.exists():
        raise SystemExit(
            f"{vecnorm} not found — its VecNormalize stats are required. "
            f"(EvalCallback saves them next to best_model.zip on each new best; "
            f"the final pair is in the run root.)"
        )
    return model, vecnorm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--run", required=True, help="training run dir, e.g. runs/stage0")
    p.add_argument(
        "--source",
        choices=("best", "final"),
        default="best",
        help="which checkpoint to promote (default: eval-best)",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="frame-skip the run was trained with (recorded for the autopilot)",
    )
    p.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="tracked models directory (default: <repo>/models)",
    )
    p.add_argument(
        "--mission-supervisor",
        action="store_true",
        help="record and auto-enable the shared in-game mission supervisor",
    )
    args = p.parse_args()

    src_model, src_vecnorm = _source_pair(Path(args.run), args.source)

    dest_dir = Path(args.models_dir) / args.stage
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_model = dest_dir / "model.zip"
    dest_vecnorm = dest_dir / "vecnorm.pkl"
    shutil.copy2(src_model, dest_model)
    shutil.copy2(src_vecnorm, dest_vecnorm)
    print(f"Copied {src_model} -> {dest_model}")
    print(f"Copied {src_vecnorm} -> {dest_vecnorm}")

    onnx, norm = export_policy(
        dest_model,
        dest_vecnorm,
        dest_dir / "model.onnx",
        stage=args.stage,
        frame_skip=args.frame_skip,
        mission_supervisor=args.mission_supervisor,
    )
    print(f"Exported {onnx}")
    print(f"Exported {norm}")
    print(f"\nPromoted {args.source} of {args.run} → {dest_dir}/")
    print("Commit the bundle to share it:")
    print(f"  git add {dest_dir} && git commit -m 'Promote {args.stage} model'")


if __name__ == "__main__":
    main()
