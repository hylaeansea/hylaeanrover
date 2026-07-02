// PyO3 0.22's macro-generated trampolines trip a couple of lints we
// can't fix in our source: edition 2024's stricter `unsafe_op_in_unsafe_fn`,
// and clippy's `useless_conversion` (the `#[pymethods]` expansion adds an
// `.into()` on the returned `PyErr`). The generated code is correct, so
// silence the noise rather than carry warnings.
#![allow(unsafe_op_in_unsafe_fn, clippy::useless_conversion)]

//! PyO3 extension module: `RoverEnv` Python class wrapping a headless
//! Bevy `App` so SB3 can drive it with `reset` / `step`.
//!
//! Architecture:
//!   * Each `RoverEnv` instance owns a `bevy::App` with `MinimalPlugins`
//!     + `RoverCorePlugin::headless()` — no window, rendering, or glTF.
//!   * `step(action)` writes `RoverAction` into the app, advances the
//!     schedule once via `app.update()`, then reads the resulting
//!     `RoverTelemetry`, `RewardState`, and `GameState` back out.
//!   * The observation comes out as a `Vec<f32>` Python sees as a list
//!     (or wrapped numpy array on the Python side).
//!
//! Time is advanced under our control via
//! `TimeUpdateStrategy::ManualDuration` so each `step()` is one fixed
//! dt regardless of wall clock — required for reproducible RL training.

use std::cell::RefCell;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::asset::AssetPlugin;
use bevy::diagnostic::DiagnosticsPlugin;
use bevy::input::InputPlugin;
use bevy::mesh::Mesh;
use bevy::pbr::StandardMaterial;
use bevy::prelude::*;
use bevy::scene::ScenePlugin;
use bevy::time::TimeUpdateStrategy;
use bevy::transform::TransformPlugin;

use hylaeanrover_core::game_state::{GameOverReason, GameState, GameStatus};
use hylaeanrover_core::power_cubes::{PowerState, RelaunchEvent};
use hylaeanrover_core::reward::{BEACON_BUDGET, RewardState};
use hylaeanrover_core::telemetry::RoverTelemetry;
use hylaeanrover_core::terrain_controls::TerrainState;
use hylaeanrover_core::{ChassisEntity, RoverAction, RoverCoreConfig, RoverCorePlugin};

use pyo3::prelude::*;

/// Fixed timestep each `step()` advances. 1/60 s matches the game's
/// rendered framerate so physics behaviour is identical across env
/// and game.
const FIXED_DT: f32 = 1.0 / 60.0;

/// Observation layout — see `hylaeanrover_core::observation` for the
/// authoritative slot map. The builder lives in the core crate so the
/// headless env and the in-game autopilot feed the policy *identical*
/// vectors.
const OBS_DIM: usize = hylaeanrover_core::observation::OBS_DIM;

/// Bevy's `App` is `!Send`, so we mark the pyclass `unsendable` —
/// Python can only access the env from the thread that created it.
/// That's fine for our use (SB3 holds the env on one thread). The
/// `RefCell` gives us interior mutability through PyO3's `&self`.
#[pyclass(unsendable)]
struct RoverEnv {
    inner: RefCell<EnvInner>,
}

struct EnvInner {
    app: App,
    seed: u64,
    step_count: u32,
    max_steps: u32,
    /// `RewardState.total()` from the previous step, so we can compute
    /// the per-step reward delta SB3 wants.
    last_total_reward: f32,
}

impl EnvInner {
    fn new(seed: u64, max_steps: u32, beacons_enabled: bool, power_capacity: Option<f32>) -> Self {
        let mut app = App::new();

        // MinimalPlugins gives us TaskPool / Time / Schedule infra
        // without spinning up winit / wgpu. The added plugins below
        // are what RoverCorePlugin's dependencies actually need.
        app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(
            // Effectively "manual": ScheduleRunnerPlugin would otherwise
            // try to drive the loop itself. We override with our own
            // `update()` calls in `step()`.
            Duration::from_secs_f64(1e9),
        )))
        .add_plugins(TransformPlugin)
        .add_plugins(InputPlugin)
        .add_plugins(DiagnosticsPlugin)
        // AssetPlugin + asset type registration is cheaper than making
        // every Asset-using system Option-safe. The core's systems
        // still create Mesh / StandardMaterial assets — they just
        // never get rendered. Cheap to keep around.
        .add_plugins(AssetPlugin::default())
        // ScenePlugin gives us the SceneSpawner resource that
        // bevy_rapier's `init_async_scene_colliders` system queries
        // (even though we never actually spawn a Scene in headless).
        .add_plugins(ScenePlugin)
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        // Fixed per-frame time delta so physics is reproducible.
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            FIXED_DT as f64,
        )));

        // Headless core: no UI, primitive rover spawn (no glTF). The
        // beacon toggle lets the RL curriculum's locomotion / mineral
        // stages neutralize action index 9 (see RoverCoreConfig).
        let mut core_cfg = RoverCoreConfig::headless();
        core_cfg.beacons_enabled = beacons_enabled;
        if let Some(wh) = power_capacity {
            core_cfg.power_capacity_wh = wh;
        }
        app.add_plugins(RoverCorePlugin(core_cfg));

        // Seed the action resource so the drive system always finds it.
        app.insert_resource(RoverAction::default());

        // Override the default terrain seed BEFORE the terrain setup
        // system runs. TerrainState is init_resource'd by the controls
        // plugin with default seed=42; setup_terrain reads
        // LunarTerrainConfig::default() instead. Easiest: just rely on
        // the existing default, and let the python side re-roll via
        // the `seed=` argument in `reset()` — by reaching into
        // TerrainState.seed and forcing a regenerate.
        //
        // For now, store the requested seed so reset() can apply it.
        let _ = seed;

        Self {
            app,
            seed,
            step_count: 0,
            max_steps,
            last_total_reward: 0.0,
        }
    }

    /// Apply a discrete action [0..9].
    ///   0..8 are a 3 (throttle: -1, 0, +1) × 3 (steer: -1, 0, +1) grid.
    ///   9    is "drop beacon, no movement".
    fn apply_action(&mut self, action: u32) {
        let mut a = RoverAction::default();
        if action == 9 {
            a.drop_beacon = true;
        } else if action < 9 {
            let throttle_idx = (action / 3) as i32; // 0, 1, 2
            let steer_idx = (action % 3) as i32;
            a.throttle = (throttle_idx - 1) as f32; // -1, 0, +1
            a.steering = (steer_idx - 1) as f32;
        }
        self.app.insert_resource(a);
    }

    fn observation(&self) -> Vec<f32> {
        let world = self.app.world();
        let telem = world.resource::<RoverTelemetry>();
        let reward = world.resource::<RewardState>();
        let power = world.resource::<PowerState>();
        let maps = world.resource::<hylaeanrover_core::minerals::MineralMaps>();

        // Chassis world position for mineral sampling (None until spawn).
        let chassis_pos = world
            .resource::<ChassisEntity>()
            .0
            .and_then(|id| world.get::<GlobalTransform>(id))
            .map(|gxf| gxf.translation());

        hylaeanrover_core::observation::build_observation(telem, power, reward, chassis_pos, maps)
    }

    fn is_terminated(&self) -> bool {
        !matches!(
            self.app.world().resource::<GameState>().status,
            GameStatus::Playing
        )
    }

    fn game_over_str(&self) -> Option<&'static str> {
        match self.app.world().resource::<GameState>().status {
            GameStatus::Playing => None,
            GameStatus::GameOver(GameOverReason::OutOfPower) => Some("out_of_power"),
            GameStatus::GameOver(GameOverReason::Flipped) => Some("flipped"),
            GameStatus::GameOver(GameOverReason::BeaconsDeployed) => Some("beacons_deployed"),
        }
    }

    /// Tick the app until the rover entity has spawned + had its
    /// colliders/joints attached. Bevy needs several frames for the
    /// initial settle (terrain spawn → mineral generation → rover
    /// primitives → attach_colliders → attach_joints).
    fn warm_up(&mut self) {
        // Generous upper bound. Real settle is ≈10 frames.
        for _ in 0..120 {
            self.app.update();
            if self.app.world().resource::<ChassisEntity>().0.is_some() {
                // One more update so the joints get attached.
                self.app.update();
                return;
            }
        }
    }
}

#[pymethods]
impl RoverEnv {
    /// Create a fresh environment. `seed` controls terrain + mineral
    /// generation. `max_steps` caps an episode before SB3 calls it a
    /// `truncated` rollout. `beacons_enabled` (default `true`) gates the
    /// beacon action: set it `false` in the RL curriculum's locomotion /
    /// mineral stages so action index 9 is an inert no-op and the
    /// `beacons_deployed` game-over never fires. `power_capacity` (Wh)
    /// overrides the game's 1 kWh battery — the RL curriculum passes a
    /// small value so the power budget binds within one episode. The
    /// battery refills on every `reset()`.
    #[new]
    #[pyo3(signature = (seed = 42, max_steps = 2000, beacons_enabled = true, power_capacity = None))]
    fn new(
        seed: u64,
        max_steps: u32,
        beacons_enabled: bool,
        power_capacity: Option<f32>,
    ) -> PyResult<Self> {
        if let Some(wh) = power_capacity
            && (!wh.is_finite() || wh <= 0.0)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "power_capacity must be a positive number of Wh",
            ));
        }
        let mut inner = EnvInner::new(seed, max_steps, beacons_enabled, power_capacity);
        // Initial warm-up so the first `obs` returned from `reset()`
        // is non-trivial.
        inner.warm_up();
        // Apply the requested seed by setting TerrainState.seed and
        // firing a fake user "Randomize" — but we just write it
        // directly. The mineral regen system will catch up next
        // frame.
        if let Some(mut terrain) = inner.app.world_mut().get_resource_mut::<TerrainState>() {
            terrain.seed = seed;
        }
        // Drain a few more updates for the seed change to propagate
        // through terrain rebuild + mineral regen.
        for _ in 0..5 {
            inner.app.update();
        }
        inner.last_total_reward = inner.app.world().resource::<RewardState>().total();
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Reset the episode. Mirrors gym's `reset` contract: returns
    /// `(observation, info_dict_as_json_string)`.
    #[pyo3(signature = (seed = None))]
    fn reset(&self, _py: Python<'_>, seed: Option<u64>) -> PyResult<(Vec<f32>, String)> {
        let mut inner = self.inner.borrow_mut();

        // Send a relaunch event — the existing reward/game_state/beacon/
        // power systems all listen for this and reset cleanly.
        inner.app.world_mut().write_message(RelaunchEvent);

        // Clear the previous episode's action, or its throttle keeps
        // driving the wheels through the settle updates below — draining
        // the freshly reset battery before the episode even starts.
        inner.app.insert_resource(RoverAction::default());

        if let Some(s) = seed {
            inner.seed = s;
            if let Some(mut terrain) = inner.app.world_mut().get_resource_mut::<TerrainState>() {
                terrain.seed = s;
            }
        }

        // Let the relaunch propagate (despawn + respawn rover, reset
        // reward, reset game state, regen minerals if seed changed).
        for _ in 0..30 {
            inner.app.update();
        }
        // Make sure the rover is back.
        inner.warm_up();

        inner.step_count = 0;
        inner.last_total_reward = inner.app.world().resource::<RewardState>().total();

        let obs = inner.observation();
        let info = make_info(&inner);
        Ok((obs, info))
    }

    /// Advance one step. Returns `(obs, reward, terminated, truncated, info_json)`.
    fn step(&self, _py: Python<'_>, action: u32) -> PyResult<(Vec<f32>, f32, bool, bool, String)> {
        let mut inner = self.inner.borrow_mut();

        inner.apply_action(action);
        inner.app.update();
        inner.step_count += 1;

        let total = inner.app.world().resource::<RewardState>().total();
        let step_reward = total - inner.last_total_reward;
        inner.last_total_reward = total;

        let terminated = inner.is_terminated();
        let truncated = !terminated && inner.step_count >= inner.max_steps;

        let obs = inner.observation();
        let info = make_info(&inner);
        Ok((obs, step_reward, terminated, truncated, info))
    }

    /// Dimensionality of the observation vector. Matches OBS_DIM.
    #[staticmethod]
    fn obs_dim() -> usize {
        OBS_DIM
    }

    /// Number of discrete actions.
    #[staticmethod]
    fn action_count() -> u32 {
        10
    }

    /// Fixed timestep per step() call (seconds).
    #[staticmethod]
    fn fixed_dt() -> f32 {
        FIXED_DT
    }

    /// Beacon budget for a fresh episode — Python uses this to scale
    /// rendered HUDs that mirror the in-game readout.
    #[staticmethod]
    fn beacon_budget() -> u32 {
        BEACON_BUDGET
    }
}

fn make_info(inner: &EnvInner) -> String {
    // The same JSON the in-game bottom bar shows. Useful for debugging
    // failed rollouts from Python.
    let world = inner.app.world();
    let mut map = serde_json::Map::new();
    map.insert("step".into(), inner.step_count.into());
    map.insert("seed".into(), inner.seed.into());
    map.insert(
        "game_over".into(),
        inner
            .game_over_str()
            .map(|s| serde_json::Value::String(s.into()))
            .unwrap_or(serde_json::Value::Null),
    );
    let reward = world.resource::<RewardState>();
    map.insert(
        "reward_total".into(),
        serde_json::Value::from(reward.total()),
    );
    map.insert(
        "reward_distance".into(),
        serde_json::Value::from(reward.distance),
    );
    map.insert(
        "reward_mineral_integral".into(),
        serde_json::Value::from(reward.mineral_integral),
    );
    map.insert(
        "reward_beacon_bonus".into(),
        serde_json::Value::from(reward.beacon_bonus),
    );
    map.insert(
        "beacons_remaining".into(),
        serde_json::Value::from(reward.beacons_remaining),
    );
    // Battery fraction remaining — lets training callbacks log how much
    // of the power budget episodes actually use (and confirm resets
    // refill it).
    let power = world.resource::<PowerState>();
    map.insert(
        "power_frac".into(),
        serde_json::Value::from(if power.max > 0.0 {
            power.current / power.max
        } else {
            0.0
        }),
    );
    serde_json::Value::Object(map).to_string()
}

/// PyO3's module name must match maturin's `module-name` config in
/// `pyproject.toml`. We expose the extension as `hylaeanrover._native`
/// — an internal submodule that the public `hylaeanrover/__init__.py`
/// wraps with the gymnasium API.
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RoverEnv>()?;
    Ok(())
}
