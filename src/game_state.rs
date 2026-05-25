//! Global game status + the two end-of-run triggers.
//!
//! End conditions ("the rover can no longer make progress"):
//!   - **OutOfPower**: `PowerState.current <= 0` AND the rover has been
//!     at rest for `AT_REST_REQUIRED_SECS`.
//!   - **Flipped**: |pitch| or |roll| > `FLIP_ANGLE_DEG` AND at rest for
//!     `AT_REST_REQUIRED_SECS`. Requiring the rest avoids ending the
//!     game mid-tumble — a rover that flips over but keeps rolling
//!     might land back on its wheels.
//!
//! "At rest" means both linear and angular velocity are below their
//! thresholds; otherwise the at-rest timer resets to zero each frame.
//!
//! `power_cubes::update_game_over_visibility` reads `GameState.status`
//! to drive the existing game-over modal; this module just decides
//! *when* to flip the status and *why*.

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use crate::power_cubes::{PowerState, RelaunchEvent};
use crate::ChassisEntity;

// ---- Tuning constants ---------------------------------------------------

/// |linvel| below this counts as "stopped" for the at-rest timer.
const AT_REST_LIN_THRESHOLD: f32 = 0.1;
/// |angvel| below this also required — a wheel-spinning-in-place rover
/// is not really at rest.
const AT_REST_ANG_THRESHOLD: f32 = 0.3;
/// Continuous at-rest seconds before either game-over condition fires.
const AT_REST_REQUIRED_SECS: f32 = 5.0;
/// Pitch / roll magnitude (deg) past which the rover counts as flipped.
/// Combined with the at-rest gate, recoverable tumbles don't trigger.
const FLIP_ANGLE_DEG: f32 = 100.0;

// ---- Resource ----------------------------------------------------------

#[derive(Resource, Default)]
pub struct GameState {
    pub status: GameStatus,
    /// Seconds the chassis has continuously satisfied the "at rest"
    /// condition. Reset to 0 whenever it doesn't.
    pub at_rest_secs: f32,
}

#[derive(Default, Clone, Copy, PartialEq)]
pub enum GameStatus {
    #[default]
    Playing,
    GameOver(GameOverReason),
}

#[derive(Clone, Copy, PartialEq)]
pub enum GameOverReason {
    OutOfPower,
    Flipped,
}

impl GameOverReason {
    /// Headline shown in the game-over modal.
    pub fn headline(self) -> &'static str {
        match self {
            GameOverReason::OutOfPower => "OUT OF POWER",
            GameOverReason::Flipped => "FLIPPED",
        }
    }
    /// One-line subtitle below the headline.
    pub fn subtitle(self) -> &'static str {
        match self {
            GameOverReason::OutOfPower => "the rover has run dry.",
            GameOverReason::Flipped => "the rover is on its head.",
        }
    }
}

// ---- Plugin -----------------------------------------------------------

pub struct GameStatePlugin;

impl Plugin for GameStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GameState>()
            .add_systems(Update, (update_game_state, reset_on_relaunch));
    }
}

// ---- Systems ----------------------------------------------------------

fn update_game_state(
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<(&GlobalTransform, Option<&Velocity>)>,
    power: Res<PowerState>,
    time: Res<Time>,
    mut state: ResMut<GameState>,
) {
    // Once we've fired a game-over, freeze — the at-rest timer no longer
    // matters and we don't want to silently switch reasons mid-modal.
    if !matches!(state.status, GameStatus::Playing) {
        return;
    }

    let Some(chassis_id) = chassis_res.0 else {
        // Rover hasn't spawned yet (or is mid-respawn): drop the timer
        // so a fresh game starts with a clean slate.
        state.at_rest_secs = 0.0;
        return;
    };
    let Ok((gxf, vel)) = chassis_q.get(chassis_id) else {
        state.at_rest_secs = 0.0;
        return;
    };

    let linvel = vel.map(|v| v.linvel.length()).unwrap_or(0.0);
    let angvel = vel.map(|v| v.angvel.length()).unwrap_or(0.0);
    let at_rest = linvel < AT_REST_LIN_THRESHOLD && angvel < AT_REST_ANG_THRESHOLD;
    if at_rest {
        state.at_rest_secs += time.delta_secs();
    } else {
        state.at_rest_secs = 0.0;
    }

    // Compute the pitch / roll the same way `sync_imu_motion` does, so
    // the trigger lines up with what the IMU panel shows.
    let rot = gxf.rotation();
    let fwd = rot * Vec3::NEG_X;
    let pitch_deg = fwd.y.clamp(-1.0, 1.0).asin().to_degrees();
    let chassis_left = rot * Vec3::Z;
    let chassis_up = rot * Vec3::Y;
    let roll_deg = chassis_left.y.atan2(chassis_up.y).to_degrees();

    let flipped = pitch_deg.abs() > FLIP_ANGLE_DEG || roll_deg.abs() > FLIP_ANGLE_DEG;
    let out_of_power = power.current <= 0.0;

    if state.at_rest_secs >= AT_REST_REQUIRED_SECS {
        // Tie-breaker: flipped is reported even if power happens to be
        // out too, since "flipped at rest" is the more immediate
        // explanation for why the rover stopped.
        if flipped {
            state.status = GameStatus::GameOver(GameOverReason::Flipped);
        } else if out_of_power {
            state.status = GameStatus::GameOver(GameOverReason::OutOfPower);
        }
    }
}

fn reset_on_relaunch(
    mut events: MessageReader<RelaunchEvent>,
    mut state: ResMut<GameState>,
) {
    if events.read().count() == 0 {
        return;
    }
    *state = GameState::default();
}
