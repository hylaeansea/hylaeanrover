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
use hylaeanrover_core::mission_supervisor::{
    MissionSupervisor, MissionSupervisorConfig, SupervisorMode,
};
use hylaeanrover_core::observation::{OBS_DIM, build_observation};
use hylaeanrover_core::power_cubes::PowerState;
use hylaeanrover_core::reward::RewardState;
use hylaeanrover_core::telemetry::RoverTelemetry;
use hylaeanrover_core::{AutopilotActive, ChassisEntity, RoverAction, RoverCoreConfig};

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
    /// Optional shared mission supervisor. Enabled explicitly with
    /// `--mission-supervisor` so existing exported policies keep their
    /// previous behavior unless the operator opts into the safety layer.
    supervisor: Option<MissionSupervisor>,
    last_supervisor_mode: Option<SupervisorMode>,
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
                    "Autopilot loaded from {}{} Keys: P = toggle, O = reload.",
                    onnx_path.display(),
                    if runtime.supervisor.is_some() {
                        " with mission supervisor."
                    } else {
                        "."
                    }
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

fn mission_supervisor_cli_enabled() -> bool {
    std::env::args().any(|arg| arg == "--mission-supervisor")
}

/// The normalization sidecar lives next to the ONNX file with the
/// extension swapped to `.norm.json` (e.g. `model.onnx` →
/// `model.norm.json`).
fn norm_path_for(onnx_path: &Path) -> PathBuf {
    onnx_path.with_extension("norm.json")
}

/// If `--policy <path>` is present and its sidecar `.norm.json` carries
/// the exported `beacons_enabled` / runtime power capacity (see
/// `export.py`), apply them on top of `default` so the game replays the
/// checkpoint under the same conditions it trained in. Must run *before*
/// `RoverCorePlugin` is constructed (that's where these fields get baked
/// in), so this is plain pre-Bevy-App code, not a system — called from
/// `main()`.
///
/// Deliberately does *not* carry over `cube_spawn_lambda` /
/// `cube_spawn_extent`, even though `export.py` used to also write them:
/// those calibrate a Poisson spawn *rate* for a bounded ~33s training
/// episode that gets fully wiped on every `RelaunchEvent`. The game has
/// no such episode boundary — the autopilot just keeps driving
/// indefinitely — so applying a training-dense rate to an unbounded
/// session piles up cubes without limit the
/// longer you watch, instead of settling at a steady state. Battery size
/// and the beacon toggle don't have that failure mode (fixed capacity /
/// a simple boolean), so they're safe to carry over; a spawn *rate*
/// isn't. `spawn_power_cubes`'s `MAX_ALIVE_CUBES` cap is the belt-and-
/// suspenders backstop for this regardless of what rate is configured.
///
/// Falls back to `default` untouched if there's no `--policy`, no
/// sidecar file, or a sidecar missing these keys (both optional, same
/// convention `load_norm` already uses for `frame_skip`) — so existing
/// exported bundles keep working exactly as before.
pub(crate) fn resolve_core_config(default: RoverCoreConfig) -> RoverCoreConfig {
    let Some(onnx_path) = parse_policy_arg() else {
        return default;
    };
    let norm_path = norm_path_for(&onnx_path);
    let Ok(text) = std::fs::read_to_string(&norm_path) else {
        return default;
    };
    let cfg = apply_norm_overrides(default, &text);

    // Runs before DefaultPlugins (and its LogPlugin) is added, so `info!`
    // isn't guaranteed to go anywhere yet — plain eprintln instead.
    if cfg.beacons_enabled != default.beacons_enabled
        || cfg.power_capacity_wh != default.power_capacity_wh
    {
        eprintln!(
            "Autopilot: applying runtime config from {} (beacons_enabled={}, power_capacity_wh={})",
            norm_path.display(),
            cfg.beacons_enabled,
            cfg.power_capacity_wh
        );
    }
    cfg
}

/// Parse `norm_json` and apply whichever of `beacons_enabled` /
/// runtime power capacity it contains on top of `default` (see
/// `resolve_core_config` for why `cube_spawn_lambda`/`cube_spawn_extent`
/// are deliberately *not* read here even if an older or newer sidecar
/// happens to contain them). Split out from `resolve_core_config` so the
/// override logic is testable without faking `--policy` / real files on
/// disk (see the unit tests below). Malformed JSON or a missing key
/// simply leaves the corresponding field(s) at `default`.
fn apply_norm_overrides(default: RoverCoreConfig, norm_json: &str) -> RoverCoreConfig {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(norm_json) else {
        return default;
    };
    let mut cfg = default;
    if let Some(b) = v["beacons_enabled"].as_bool() {
        cfg.beacons_enabled = b;
    }
    if let Some(wh) = v["runtime_power_capacity_wh"]
        .as_f64()
        .or_else(|| v["power_capacity_wh"].as_f64())
    {
        cfg.power_capacity_wh = wh as f32;
    }
    cfg
}

fn load_runtime(onnx_path: &Path) -> TractResult<AutopilotRuntime> {
    let policy = load_policy(onnx_path)?;
    let (norm, frame_skip, sidecar_supervisor, supervisor_config) =
        load_norm(&norm_path_for(onnx_path))?;
    Ok(AutopilotRuntime {
        policy,
        norm,
        onnx_path: onnx_path.to_path_buf(),
        frame_skip: frame_skip.max(1),
        tick: 0,
        held: RoverAction::default(),
        supervisor: (mission_supervisor_cli_enabled() || sidecar_supervisor)
            .then(|| MissionSupervisor::new(supervisor_config)),
        last_supervisor_mode: None,
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
fn load_norm(path: &Path) -> TractResult<(NormStats, u32, bool, MissionSupervisorConfig)> {
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
    let (mission_supervisor, beacons_enabled) = sidecar_runtime_flags(&v);
    let supervisor_config = supervisor_config_from_sidecar(&v, beacons_enabled);
    Ok((
        NormStats {
            mean,
            var,
            clip,
            epsilon,
        },
        frame_skip,
        mission_supervisor,
        supervisor_config,
    ))
}

fn sidecar_runtime_flags(v: &serde_json::Value) -> (bool, bool) {
    (
        v["mission_supervisor"].as_bool().unwrap_or(false),
        v["beacons_enabled"].as_bool().unwrap_or(true),
    )
}

fn supervisor_config_from_sidecar(
    v: &serde_json::Value,
    beacons_enabled: bool,
) -> MissionSupervisorConfig {
    let mut config = MissionSupervisorConfig {
        beacon_guard_enabled: beacons_enabled,
        ..MissionSupervisorConfig::default()
    };
    let enter = v["supervisor_low_power_enter_fraction"]
        .as_f64()
        .map(|value| value as f32)
        .unwrap_or(config.low_power_enter_fraction);
    let exit = v["supervisor_low_power_exit_fraction"]
        .as_f64()
        .map(|value| value as f32)
        .unwrap_or(config.low_power_exit_fraction);
    if enter.is_finite() && exit.is_finite() && 0.0 <= enter && enter < exit && exit <= 1.0 {
        config.low_power_enter_fraction = enter;
        config.low_power_exit_fraction = exit;
    }
    config
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
            if let Some(supervisor) = runtime.supervisor.as_mut() {
                supervisor.reset();
            }
            runtime.last_supervisor_mode = None;
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
        let policy_obs = runtime.supervisor.as_ref().map_or_else(
            || obs.clone(),
            |supervisor| supervisor.policy_observation(&obs),
        );
        let normalized = runtime.norm.normalize(&policy_obs);
        match infer(&runtime.policy, &normalized) {
            Ok(policy_index) => {
                let index = if let Some(supervisor) = runtime.supervisor.as_mut() {
                    let decision = supervisor.decide_with_context(
                        &obs,
                        policy_index as u32,
                        power.max,
                        reward.distance,
                    );
                    if runtime.last_supervisor_mode != Some(decision.mode) {
                        info!(
                            "Mission supervisor: mode={} policy_action={} action={} target_visible={} target_viable={} available_range={:.1}m",
                            decision.mode.as_str(),
                            decision.proposed_action,
                            decision.action,
                            decision.target_visible,
                            decision.target_viable,
                            decision.available_range_m,
                        );
                        runtime.last_supervisor_mode = Some(decision.mode);
                    }
                    decision.action as usize
                } else {
                    policy_index
                };
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

#[cfg(test)]
mod tests {
    use super::*;

    // A stage trained with beacons disabled and a small battery (e.g.
    // power_cubes) should override both fields on top of the game's
    // defaults — this is the exact scenario that let a policy silently
    // drop live beacons in the game (see git history). `cube_spawn_*`
    // must stay at the game's default even though this (older-format)
    // sidecar contains them — carrying over a training-dense spawn rate
    // into the game's unbounded session is what caused cubes to pile up
    // without limit (see git history, again).
    #[test]
    fn applies_beacons_and_power_but_not_cube_spawn() {
        let default = RoverCoreConfig::default();
        let norm_json = r#"{
            "beacons_enabled": false,
            "power_capacity_wh": 100.0,
            "cube_spawn_lambda": 3.5,
            "cube_spawn_extent": 25.0
        }"#;
        let cfg = apply_norm_overrides(default, norm_json);
        assert!(!cfg.beacons_enabled);
        assert_eq!(cfg.power_capacity_wh, 100.0);
        assert_eq!(cfg.cube_spawn_lambda, default.cube_spawn_lambda);
        assert_eq!(cfg.cube_spawn_extent, default.cube_spawn_extent);
    }

    #[test]
    fn runtime_power_capacity_overrides_training_and_legacy_values() {
        let default = RoverCoreConfig::default();
        let norm_json = r#"{
            "power_capacity_wh": 100.0,
            "training_power_capacity_wh": 100.0,
            "runtime_power_capacity_wh": 1000.0
        }"#;
        let cfg = apply_norm_overrides(default, norm_json);
        assert_eq!(cfg.power_capacity_wh, 1000.0);
    }

    // Old-format sidecars (exported before this fix) lack these keys
    // entirely — must not regress existing bundles like `models/locomotion`.
    #[test]
    fn missing_keys_fall_back_to_default() {
        let default = RoverCoreConfig::default();
        let norm_json = r#"{"mean": [], "var": [], "frame_skip": 4}"#;
        let cfg = apply_norm_overrides(default, norm_json);
        assert_eq!(cfg.beacons_enabled, default.beacons_enabled);
        assert_eq!(cfg.power_capacity_wh, default.power_capacity_wh);
    }

    // Malformed JSON must not panic — just fall back entirely.
    #[test]
    fn malformed_json_falls_back_to_default() {
        let default = RoverCoreConfig::default();
        let cfg = apply_norm_overrides(default, "not json");
        assert_eq!(cfg.beacons_enabled, default.beacons_enabled);
        assert_eq!(cfg.power_capacity_wh, default.power_capacity_wh);
    }

    // `full`-stage exports have beacons_enabled=true, matching the game's
    // own default — should be a no-op (other than the harmless re-set).
    #[test]
    fn full_stage_matches_game_defaults() {
        let default = RoverCoreConfig::default();
        let norm_json = r#"{
            "beacons_enabled": true,
            "power_capacity_wh": 100.0,
            "cube_spawn_lambda": 3.5,
            "cube_spawn_extent": 25.0
        }"#;
        let cfg = apply_norm_overrides(default, norm_json);
        assert_eq!(cfg.beacons_enabled, default.beacons_enabled);
    }

    #[test]
    fn sidecar_flags_enable_supervisor_and_preserve_stage_beacon_mode() {
        let minerals: serde_json::Value =
            serde_json::from_str(r#"{"mission_supervisor": true, "beacons_enabled": false}"#)
                .unwrap();
        assert_eq!(sidecar_runtime_flags(&minerals), (true, false));

        let legacy: serde_json::Value = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(sidecar_runtime_flags(&legacy), (false, true));
    }

    #[test]
    fn sidecar_applies_runtime_supervisor_power_thresholds() {
        let sidecar: serde_json::Value = serde_json::from_str(
            r#"{
                "beacons_enabled": true,
                "supervisor_low_power_enter_fraction": 0.15,
                "supervisor_low_power_exit_fraction": 0.20
            }"#,
        )
        .unwrap();
        let config = supervisor_config_from_sidecar(&sidecar, true);
        assert_eq!(config.low_power_enter_fraction, 0.15);
        assert_eq!(config.low_power_exit_fraction, 0.20);
        assert!(config.beacon_guard_enabled);
    }

    #[test]
    fn invalid_sidecar_power_thresholds_fall_back_to_training_defaults() {
        let sidecar: serde_json::Value = serde_json::from_str(
            r#"{
                "supervisor_low_power_enter_fraction": 0.40,
                "supervisor_low_power_exit_fraction": 0.15
            }"#,
        )
        .unwrap();
        let config = supervisor_config_from_sidecar(&sidecar, false);
        let defaults = MissionSupervisorConfig::default();
        assert_eq!(
            config.low_power_enter_fraction,
            defaults.low_power_enter_fraction
        );
        assert_eq!(
            config.low_power_exit_fraction,
            defaults.low_power_exit_fraction
        );
        assert!(!config.beacon_guard_enabled);
    }
}
