//! Game logic + physics for the rover simulator. This crate is shared
//! between the playable game (`hylaeanrover_game`) and the RL Python
//! extension (`hylaeanrover_py`).
//!
//! `RoverCorePlugin` is the high-level convenience plugin that registers
//! every sub-plugin in one call. Pass `RoverCoreConfig` to opt out of
//! UI / asset-dependent pieces for headless RL.

// Bevy ECS systems legitimately take many `Res`/`Query` parameters and
// the query filter types are intentionally verbose — these two clippy
// lints are noise for an ECS codebase, so silence them crate-wide.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

pub mod beacons;
pub mod game_state;
pub mod imu;
pub mod minerals;
pub mod observation;
pub mod power_cubes;
pub mod reward;
pub mod rover;
pub mod telemetry;
pub mod terrain;
pub mod terrain_controls;
pub mod ui;

pub use rover::{ChassisEntity, ROVER_GROUP, RoverAction, RoverPlugin, RoverRoot, RoverSpawnMode};

/// Configures the `RoverCorePlugin` bundle.
#[derive(Clone, Copy)]
pub struct RoverCoreConfig {
    /// `Gltf` (game) or `Primitives` (headless RL).
    pub spawn_mode: RoverSpawnMode,
    /// `true` for the playable game, `false` for headless. Skips
    /// UiFontPlugin + the sidebar container so panel-spawn systems
    /// no-op cleanly (each has been made `Option<Res<UiFont>>`-safe).
    pub with_ui: bool,
    /// When `false`, the beacon action (drop_beacon / B key) is a no-op
    /// and the `BeaconsDeployed` game-over never fires. Used by the RL
    /// curriculum's locomotion / mineral stages so action index 9 is an
    /// inert no-op (keeping the action space `Discrete(10)` fixed across
    /// stages) instead of silently ending the episode after 5 presses.
    pub beacons_enabled: bool,
    /// Battery capacity in Wh for a fresh run. The game keeps the
    /// 1 kWh default; the RL env shrinks it so the power budget binds
    /// within a single episode and power management becomes part of
    /// the learned behavior.
    pub power_capacity_wh: f32,
    /// Power-cube Poisson spawn rate (cubes/sec). The game and most RL
    /// stages keep the default; the `power_cubes` curriculum stage raises
    /// it so a short episode sees enough cubes to learn seek behavior.
    pub cube_spawn_lambda: f32,
    /// Half-width (m) of the square region power cubes spawn within. The
    /// game and most RL stages keep the default; the `power_cubes` stage
    /// shrinks it so denser cubes stay reachable within one episode.
    pub cube_spawn_extent: f32,
    /// Initial terrain height multiplier. The game keeps 1.0 by default;
    /// RL eval/training can vary it per reset to prove locomotion is not
    /// overfit to one terrain roughness.
    pub terrain_height_scale: f32,
    /// Seed for power-cube spawn randomness. The RL env reseeds this on
    /// every reset so cube spawning is reproducible under the Gym seed.
    pub cube_spawn_seed: u64,
}

impl Default for RoverCoreConfig {
    fn default() -> Self {
        Self {
            spawn_mode: RoverSpawnMode::Gltf,
            with_ui: true,
            beacons_enabled: true,
            power_capacity_wh: power_cubes::POWER_MAX,
            cube_spawn_lambda: power_cubes::SPAWN_LAMBDA,
            cube_spawn_extent: power_cubes::SPAWN_EXTENT,
            terrain_height_scale: 1.0,
            cube_spawn_seed: 42,
        }
    }
}

impl RoverCoreConfig {
    pub fn headless() -> Self {
        Self {
            spawn_mode: RoverSpawnMode::Primitives,
            with_ui: false,
            beacons_enabled: true,
            power_capacity_wh: power_cubes::POWER_MAX,
            cube_spawn_lambda: power_cubes::SPAWN_LAMBDA,
            cube_spawn_extent: power_cubes::SPAWN_EXTENT,
            terrain_height_scale: 1.0,
            cube_spawn_seed: 42,
        }
    }
}

/// Resource mirror of [`RoverCoreConfig::beacons_enabled`] so the beacon
/// placement + game-over systems can read it. Defaults to enabled so the
/// game binary and any standalone test behave normally.
#[derive(Resource, Clone, Copy)]
pub struct BeaconsEnabled(pub bool);

impl Default for BeaconsEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// When present and `true`, the rover `drive` system always reads the
/// `RoverAction` resource (even when it commands zero throttle/steer),
/// instead of falling back to keyboard input. The in-game autopilot sets
/// this so a policy that chooses "coast" / "stop" isn't silently
/// overridden by the keyboard fallback. Absent / `false` keeps the
/// game's normal keyboard-driven behaviour.
#[derive(Resource, Clone, Copy, Default)]
pub struct AutopilotActive(pub bool);

/// Convenience plugin that wires up every game system the rover needs.
#[derive(Default)]
pub struct RoverCorePlugin(pub RoverCoreConfig);

impl Plugin for RoverCorePlugin {
    fn build(&self, app: &mut App) {
        // Physics. Both modes need it.
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default());

        // Mirror the beacon toggle into a resource the beacon + game-over
        // systems read.
        app.insert_resource(BeaconsEnabled(self.0.beacons_enabled));

        // UI font + sidebar container. Headless skips these entirely;
        // panel-spawn systems all check `Option<Res<UiFont>>` and
        // gracefully no-op when missing.
        if self.0.with_ui {
            app.add_plugins(ui::UiFontPlugin);
        }

        // Game logic — all of these are headless-safe (UI is gated on
        // UiFont presence inside each plugin).
        app.add_plugins(terrain_controls::TerrainControlsPlugin {
            initial_height_scale: self.0.terrain_height_scale,
        })
        .add_plugins(power_cubes::PowerCubesPlugin {
            capacity_wh: self.0.power_capacity_wh,
            spawn_lambda: self.0.cube_spawn_lambda,
            spawn_extent: self.0.cube_spawn_extent,
            rng_seed: self.0.cube_spawn_seed,
        })
        .add_plugins(beacons::BeaconsPlugin)
        .add_plugins(minerals::MineralsPlugin)
        .add_plugins(imu::ImuPlugin)
        .add_plugins(reward::RewardPlugin)
        .add_plugins(game_state::GameStatePlugin)
        .add_plugins(telemetry::TelemetryPlugin)
        .add_plugins(RoverPlugin {
            spawn_mode: self.0.spawn_mode,
        });
    }
}
