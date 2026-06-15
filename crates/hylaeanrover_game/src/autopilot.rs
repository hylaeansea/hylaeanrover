//! In-game autopilot: drive the rover with a policy trained by the RL
//! curriculum, inside the full rendered game (HUD, overlays, camera).
//!
//! Enable it by passing a policy exported from training:
//!
//! ```bash
//! cargo run -p hylaeanrover_game --release -- \
//!     --policy runs/stage0/model.onnx
//! ```
//!
//! The ONNX file is produced by `python/examples/export_policy.py`; a
//! sibling `*.norm.json` holds the `VecNormalize` observation statistics
//! (mean / var / clip) so we can normalize the observation exactly as
//! training did before feeding the network.
//!
//! Controls while autopilot is loaded:
//!   - **P** toggle autopilot on/off (off → keyboard takes over)
//!   - **O** reload the policy file from disk (watch training progress
//!     by re-exporting a newer checkpoint and pressing O)
//!
//! Inference runs each frame in `PreUpdate` so the resulting
//! `RoverAction` is in place before the core `drive` system reads it.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use tract_onnx::prelude::*;

use hylaeanrover_core::minerals::MineralMaps;
use hylaeanrover_core::observation::{OBS_DIM, build_observation};
use hylaeanrover_core::power_cubes::PowerState;
use hylaeanrover_core::reward::RewardState;
use hylaeanrover_core::telemetry::RoverTelemetry;
use hylaeanrover_core::{AutopilotActive, ChassisEntity, RoverAction};

/// A runnable, optimized ONNX graph mapping a `[1, OBS_DIM]` observation
/// to `[1, 10]` action logits.
type Policy = TypedRunnableModel<TypedModel>;

/// Observation normalization stats from the training run's
/// `VecNormalize`. `normalize` mirrors SB3's `normalize_obs`.
struct NormStats {
    mean: Vec<f32>,
    var: Vec<f32>,
    clip: f32,
    epsilon: f32,
}

impl NormStats {
    fn normalize(&self, obs: &[f32]) -> Vec<f32> {
        obs.iter()
            .zip(self.mean.iter())
            .zip(self.var.iter())
            .map(|((o, m), v)| {
                let n = (o - m) / (v + self.epsilon).sqrt();
                n.clamp(-self.clip, self.clip)
            })
            .collect()
    }
}

#[derive(Resource)]
struct AutopilotRuntime {
    policy: Policy,
    norm: NormStats,
    /// Path to the ONNX file, so **O** can reload it from disk.
    onnx_path: PathBuf,
    /// Hold each inferred action for this many frames (matches the
    /// `ActionRepeat`/frame-skip used at training time).
    frame_skip: u32,
    /// Frames since the last inference; a decision is made when
    /// `tick % frame_skip == 0`.
    tick: u32,
    /// The most recent inferred action, reused on held (non-decision)
    /// frames.
    held: RoverAction,
}

pub struct AutopilotPlugin;

impl Plugin for AutopilotPlugin {
    fn build(&self, app: &mut App) {
        let Some(onnx_path) = parse_policy_arg() else {
            return; // No --policy: game runs in normal manual-drive mode.
        };

        match load_runtime(&onnx_path) {
            Ok(runtime) => {
                info!(
                    "Autopilot loaded from {}. Keys: P = toggle, O = reload.",
                    onnx_path.display()
                );
                app.insert_resource(runtime)
                    .insert_resource(RoverAction::default())
                    .insert_resource(AutopilotActive(true))
                    .add_systems(PreUpdate, autopilot_drive);
            }
            Err(e) => {
                error!(
                    "Failed to load autopilot policy {}: {e}",
                    onnx_path.display()
                );
            }
        }
    }
}

/// Read `--policy <path>` from the process args.
fn parse_policy_arg() -> Option<PathBuf> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == "--policy" {
            return args.next().map(PathBuf::from);
        }
        if let Some(rest) = arg.strip_prefix("--policy=") {
            return Some(PathBuf::from(rest));
        }
    }
    None
}

/// The normalization sidecar lives next to the ONNX file with the
/// extension swapped to `.norm.json` (e.g. `model.onnx` →
/// `model.norm.json`).
fn norm_path_for(onnx_path: &Path) -> PathBuf {
    onnx_path.with_extension("norm.json")
}

fn load_runtime(onnx_path: &Path) -> TractResult<AutopilotRuntime> {
    let policy = load_policy(onnx_path)?;
    let (norm, frame_skip) = load_norm(&norm_path_for(onnx_path))?;
    Ok(AutopilotRuntime {
        policy,
        norm,
        onnx_path: onnx_path.to_path_buf(),
        frame_skip: frame_skip.max(1),
        tick: 0,
        held: RoverAction::default(),
    })
}

fn load_policy(path: &Path) -> TractResult<Policy> {
    tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, f32::fact([1, OBS_DIM]).into())?
        .into_optimized()?
        .into_runnable()
}

/// Parse the `*.norm.json` sidecar: observation normalization stats plus
/// the frame-skip the policy was trained with.
fn load_norm(path: &Path) -> TractResult<(NormStats, u32)> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;

    let read_vec = |key: &str| -> TractResult<Vec<f32>> {
        let arr = v[key].as_array().ok_or_else(|| {
            anyhow::anyhow!("`{key}` missing or not an array in {}", path.display())
        })?;
        Ok(arr
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect())
    };

    let mean = read_vec("mean")?;
    let var = read_vec("var")?;
    if mean.len() != OBS_DIM || var.len() != OBS_DIM {
        return Err(anyhow::anyhow!(
            "norm stats length {} / {} != OBS_DIM {OBS_DIM}",
            mean.len(),
            var.len()
        ));
    }
    let clip = v["clip_obs"].as_f64().unwrap_or(10.0) as f32;
    let epsilon = v["epsilon"].as_f64().unwrap_or(1e-8) as f32;
    // `frame_skip` is optional for backward compatibility (default 1).
    let frame_skip = v["frame_skip"].as_u64().unwrap_or(1) as u32;
    Ok((
        NormStats {
            mean,
            var,
            clip,
            epsilon,
        },
        frame_skip,
    ))
}

/// Map a discrete action index [0..9] to a `RoverAction`. Mirrors the
/// Python env's `apply_action` so the policy means the same thing here.
fn action_for(index: usize) -> RoverAction {
    let mut a = RoverAction::default();
    if index == 9 {
        a.drop_beacon = true;
    } else if index < 9 {
        let throttle_idx = (index / 3) as i32; // 0,1,2
        let steer_idx = (index % 3) as i32;
        a.throttle = (throttle_idx - 1) as f32; // -1,0,+1
        a.steering = (steer_idx - 1) as f32;
    }
    a
}

fn autopilot_drive(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut active: ResMut<AutopilotActive>,
    mut runtime: ResMut<AutopilotRuntime>,
    telem: Res<RoverTelemetry>,
    power: Res<PowerState>,
    reward: Res<RewardState>,
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    maps: Res<MineralMaps>,
    mut action: ResMut<RoverAction>,
) {
    // P: toggle. When turning off, zero the action so the keyboard
    // fallback (and a stopped rover) takes over cleanly. When turning on,
    // reset the frame-skip counter so the next frame is a decision frame.
    if keyboard.just_pressed(KeyCode::KeyP) {
        active.0 = !active.0;
        info!("Autopilot {}", if active.0 { "ON" } else { "OFF (manual)" });
        if active.0 {
            runtime.tick = 0;
        } else {
            *action = RoverAction::default();
        }
    }

    // O: hot-reload the policy from disk (e.g. a newer checkpoint).
    if keyboard.just_pressed(KeyCode::KeyO) {
        let path = runtime.onnx_path.clone();
        match load_runtime(&path) {
            Ok(fresh) => {
                *runtime = fresh;
                info!("Reloaded autopilot policy from {}", path.display());
            }
            Err(e) => error!("Reload failed: {e}"),
        }
    }

    if !active.0 || !telem.ready {
        return;
    }

    let frame_skip = runtime.frame_skip.max(1);
    if runtime.tick.is_multiple_of(frame_skip) {
        // Decision frame: run inference and remember the action.
        let chassis_pos = chassis_res
            .0
            .and_then(|id| xforms.get(id).ok())
            .map(|gxf| gxf.translation());
        let obs = build_observation(&telem, &power, &reward, chassis_pos, &maps);
        let normalized = runtime.norm.normalize(&obs);
        match infer(&runtime.policy, &normalized) {
            Ok(index) => {
                let a = action_for(index);
                runtime.held = a;
                *action = a;
            }
            Err(e) => error!("Autopilot inference failed: {e}"),
        }
    } else {
        // Held frame: repeat the motor command, but never re-drop a
        // beacon (it fired on the decision frame) — matches ActionRepeat.
        let mut a = runtime.held;
        a.drop_beacon = false;
        *action = a;
    }
    runtime.tick = runtime.tick.wrapping_add(1);
}

/// Run the policy and return the argmax action index.
fn infer(policy: &Policy, normalized: &[f32]) -> TractResult<usize> {
    let input = tract_ndarray::Array2::from_shape_vec((1, OBS_DIM), normalized.to_vec())?;
    let outputs = policy.run(tvec!(Tensor::from(input).into()))?;
    let logits = outputs[0].to_array_view::<f32>()?;
    let index = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(4); // 4 = stop, a safe default
    Ok(index)
}
