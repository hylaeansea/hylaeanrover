"""Export a trained SB3 PPO policy to ONNX for the in-game autopilot.

Produces, next to the chosen `.onnx` path:

  * ``<out>.onnx``       — the policy network: obs[1,42] → action logits[1,10]
  * ``<out>.norm.json``  — VecNormalize obs stats (mean/var/clip) + frame-skip
                           so the game normalizes/replays exactly as training.

Example::

    python examples/export_policy.py \
        --model runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl

Then watch it drive in the full game::

    cargo run -p hylaeanrover_game --release -- --policy runs/stage0/model.onnx

(For curating a per-stage "best" bundle under `models/`, use
`examples/promote_model.py` instead.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hylaeanrover.export import export_policy
from hylaeanrover.wrappers import STAGES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="trained model.zip")
    p.add_argument("--vecnorm", required=True, help="matching vecnorm.pkl")
    p.add_argument("--stage", choices=STAGES, default="full", help="env stage for the dummy obs")
    p.add_argument("--out", default=None, help="output .onnx path (default: alongside model)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--frame-skip", type=int, default=1,
                   help="frame-skip used in training; recorded so the autopilot matches it")
    args = p.parse_args()

    out_onnx = Path(args.out) if args.out else Path(args.model).with_suffix(".onnx")
    onnx, norm = export_policy(
        args.model, args.vecnorm, out_onnx,
        stage=args.stage, frame_skip=args.frame_skip, opset=args.opset,
    )
    print(f"Wrote {onnx}")
    print(f"Wrote {norm}")
    print(f"\nRun: cargo run -p hylaeanrover_game --release -- --policy {onnx}")


if __name__ == "__main__":
    main()
