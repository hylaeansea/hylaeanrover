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
}

impl Default for RoverCoreConfig {
    fn default() -> Self {
        Self {
            spawn_mode: RoverSpawnMode::Gltf,
            with_ui: true,
            beacons_enabled: true,
        }
    }
}

impl RoverCoreConfig {
    pub fn headless() -> Self {
        Self {
            spawn_mode: RoverSpawnMode::Primitives,
            with_ui: false,
            beacons_enabled: true,
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
        app.add_plugins(terrain_controls::TerrainControlsPlugin)
            .add_plugins(power_cubes::PowerCubesPlugin)
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
