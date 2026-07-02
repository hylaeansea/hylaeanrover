# Curated models

The best-so-far policy for each curriculum stage, committed to the repo
so anyone can resume training, run the in-game autopilot, or compare
against it. The bundles are tiny (~240 KB/stage), so plain git is fine
(no LFS needed).

## Layout

```
models/<stage>/
├── model.zip        SB3 PPO policy — resume training or re-export
├── vecnorm.pkl      matching VecNormalize observation stats
├── model.onnx       policy network for the in-game autopilot
└── model.norm.json  obs stats + frame-skip for the autopilot
```

`<stage>` is one of `locomotion`, `power_cubes`, `minerals`, `full`.

## Promoting a run into here

After a training run you're happy with (see `python/README.md`):

```bash
cd python && source .venv/bin/activate
python examples/promote_model.py --stage locomotion --run runs/stage0
# add --frame-skip K if you trained with frame-skip; --source final to
# promote the last model instead of the eval-best one.
git add ../models/locomotion && git commit -m "Promote locomotion model"
```

`promote_model.py` copies the chosen `model.zip` + `vecnorm.pkl` here and
regenerates `model.onnx` + `model.norm.json` from them.

## Using a promoted model

Resume training from the stage's best (e.g. continue locomotion, or warm-
start the next stage):

```bash
python examples/train.py --stage power_cubes --timesteps 1000000 \
    --load models/locomotion/model.zip --vecnorm models/locomotion/vecnorm.pkl \
    --save runs/power_cubes
```

Evaluate it:

```bash
python examples/evaluate.py --stage locomotion \
    --load models/locomotion/model.zip --vecnorm models/locomotion/vecnorm.pkl
```

Watch it in the game:

```bash
cargo run -p hylaeanrover_game --release -- --policy models/locomotion/model.onnx
```

## Note on portability

`model.zip` / `vecnorm.pkl` are Python pickles — resuming training needs
compatible Stable-Baselines3 / PyTorch / NumPy versions. The `model.onnx`
bundle is self-contained and version-independent, so the autopilot works
regardless.
