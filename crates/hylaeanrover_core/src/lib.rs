//! Game logic + physics for the rover simulator. This crate is shared
//! between the playable game (`hylaeanrover_game`) and the RL Python
//! extension (`hylaeanrover_py`).
//!
//! `RoverCorePlugin` is the high-level convenience plugin that registers
//! every sub-plugin in one call. Pass `RoverCoreConfig` to opt out of
//! UI / asset-dependent pieces for headless RL.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

pub mod beacons;
pub mod game_state;
pub mod imu;
pub mod minerals;
pub mod power_cubes;
pub mod reward;
pub mod rover;
pub mod telemetry;
pub mod terrain;
pub mod terrain_controls;
pub mod ui;

pub use rover::{
    ChassisEntity, RoverAction, RoverPlugin, RoverRoot, RoverSpawnMode, ROVER_GROUP,
};

/// Configures the `RoverCorePlugin` bundle.
#[derive(Clone, Copy)]
pub struct RoverCoreConfig {
    /// `Gltf` (game) or `Primitives` (headless RL).
    pub spawn_mode: RoverSpawnMode,
    /// `true` for the playable game, `false` for headless. Skips
    /// UiFontPlugin + the sidebar container so panel-spawn systems
    /// no-op cleanly (each has been made `Option<Res<UiFont>>`-safe).
    pub with_ui: bool,
}

impl Default for RoverCoreConfig {
    fn default() -> Self {
        Self { spawn_mode: RoverSpawnMode::Gltf, with_ui: true }
    }
}

impl RoverCoreConfig {
    pub fn headless() -> Self {
        Self { spawn_mode: RoverSpawnMode::Primitives, with_ui: false }
    }
}

/// Convenience plugin that wires up every game system the rover needs.
pub struct RoverCorePlugin(pub RoverCoreConfig);

impl Default for RoverCorePlugin {
    fn default() -> Self {
        Self(RoverCoreConfig::default())
    }
}

impl Plugin for RoverCorePlugin {
    fn build(&self, app: &mut App) {
        // Physics. Both modes need it.
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default());

        // UI font + sidebar container. Headless skips these entirely;
        // panel-spawn systems all check `Option<Res<UiFont>>` and
        // gracefully no-op when missing.
        if self.0.with_ui {
            app.add_plugins(ui::UiFontPlugin);
        }

        // Game logic — all of these are headless-safe (UI is gated on
        // UiFont presence inside each plugin).
        app.add_plugins(terrain_controls::TerrainControlsPlugin)
            .add_plugins(power_cubes::PowerCubesPlugin)
            .add_plugins(beacons::BeaconsPlugin)
            .add_plugins(minerals::MineralsPlugin)
            .add_plugins(imu::ImuPlugin)
            .add_plugins(reward::RewardPlugin)
            .add_plugins(game_state::GameStatePlugin)
            .add_plugins(telemetry::TelemetryPlugin)
            .add_plugins(RoverPlugin { spawn_mode: self.0.spawn_mode });
    }
}
