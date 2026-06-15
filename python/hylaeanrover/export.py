"""Export a trained SB3 PPO policy to the ONNX bundle the game autopilot
loads. Shared by `examples/export_policy.py` and `examples/promote_model.py`.

Produces two files next to the chosen `.onnx` path:

  * ``<out>.onnx``      — the policy network: obs[1,42] → action logits[1,10]
  * ``<out>.norm.json`` — VecNormalize obs stats (mean/var/clip/epsilon)
                          plus the training frame-skip, so the game
                          normalizes and replays exactly as training did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hylaeanrover import OBS_DIM
from hylaeanrover.wrappers import make_staged_env

PathLike = Union[str, Path]


class _OnnxPolicy(th.nn.Module):
    """Wrap an SB3 policy to output raw action logits (argmax in Rust).

    Exporting just the logits keeps the graph to plain linear/activation
    ops — no sampling/argmax nodes — which `tract-onnx` runs cleanly.
    """

    def __init__(self, policy: th.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)


def export_policy(
    model_path: PathLike,
    vecnorm_path: PathLike,
    out_onnx: PathLike,
    stage: str = "full",
    frame_skip: int = 1,
    opset: int = 17,
) -> tuple[Path, Path]:
    """Export `model_path` (+ `vecnorm_path`) to `out_onnx` + sibling
    `.norm.json`. Returns the two written paths."""
    out_onnx = Path(out_onnx)
    out_norm = out_onnx.with_suffix(".norm.json")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    # --- Policy network → ONNX --------------------------------------------
    model = PPO.load(str(model_path), device="cpu")
    model.policy.eval()
    dummy = th.zeros((1, OBS_DIM), dtype=th.float32)
    # dynamo=False uses the legacy TorchScript exporter: no `onnxscript`
    # dependency, and it emits the plain Gemm/activation ops tract reads
    # cleanly. (The default dynamo exporter needs extra packages.)
    th.onnx.export(
        _OnnxPolicy(model.policy),
        dummy,
        str(out_onnx),
        input_names=["obs"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
    )

    # --- VecNormalize obs stats + frame-skip → JSON sidecar ---------------
    venv = DummyVecEnv([lambda: make_staged_env(stage)])
    vn = VecNormalize.load(str(vecnorm_path), venv)
    stats = {
        "mean": vn.obs_rms.mean.astype(float).tolist(),
        "var": vn.obs_rms.var.astype(float).tolist(),
        "clip_obs": float(vn.clip_obs),
        "epsilon": float(vn.epsilon),
        # The in-game autopilot holds each action for this many physics
        # ticks, matching ActionRepeat at training time.
        "frame_skip": int(frame_skip),
    }
    venv.close()
    if len(stats["mean"]) != OBS_DIM:
        raise SystemExit(f"vecnorm obs dim {len(stats['mean'])} != OBS_DIM {OBS_DIM}")
    out_norm.write_text(json.dumps(stats))

    return out_onnx, out_norm
