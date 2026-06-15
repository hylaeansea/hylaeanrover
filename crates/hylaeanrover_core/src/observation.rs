//! The RL observation vector, built in one place so the headless Python
//! env (`hylaeanrover_py`) and the in-game autopilot produce *identical*
//! inputs for the policy network.
//!
//! Layout (42 floats):
//!   - 7  IMU: speed, heading, pitch, roll, yaw rate, accel fwd, accel lat
//!   - 8  lidar fan, meters (200 = no hit)
//!   - 18 6 visible cubes × (bearing_deg, distance_m, valid_flag)
//!   - 2  power: normalized 0..1, raw Wh
//!   - 6  mineral surface concentrations under the rover (Si..He-3)
//!   - 1  beacons remaining
//!
//! Deliberately excluded: the cumulative reward breakdown and the
//! game-over flag (unbounded / non-Markovian — bad policy inputs).

use bevy::math::Vec3;

use crate::minerals::MineralMaps;
use crate::power_cubes::PowerState;
use crate::reward::RewardState;
use crate::telemetry::RoverTelemetry;

/// Total observation length. The single source of truth for both the
/// Python `Box` space and the in-game policy input.
pub const OBS_DIM: usize = 42;

/// Number of visible-cube slots in the observation (padded layout).
pub const MAX_VISIBLE_CUBES: usize = 6;

/// Build the 42-float observation vector from the simulation state.
///
/// `chassis_pos` is the chassis's world position (used to sample mineral
/// concentrations); pass `None` if the rover hasn't spawned yet, in which
/// case the mineral slots are zero-filled.
pub fn build_observation(
    telem: &RoverTelemetry,
    power: &PowerState,
    reward: &RewardState,
    chassis_pos: Option<Vec3>,
    maps: &MineralMaps,
) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBS_DIM);

    // ---- IMU (7) ----
    let imu = telem.imu;
    obs.extend([
        imu.speed_mps,
        imu.heading_deg,
        imu.pitch_deg,
        imu.roll_deg,
        imu.yaw_rate_deg_s,
        imu.accel_fwd_m_s2,
        imu.accel_lat_m_s2,
    ]);

    // ---- Lidar (8) ----
    obs.extend(telem.lidar_m.iter().copied());

    // ---- Visible cubes (6 × 3) ----
    // Padded: [bearing_deg, distance_m, valid]. `valid` = 1.0 when a cube
    // fills the slot so the agent can tell "no cube" from "cube at 0 m".
    for i in 0..MAX_VISIBLE_CUBES {
        match telem.visible_cubes.get(i) {
            Some(c) => obs.extend([c.bearing_deg, c.distance_m, 1.0]),
            None => obs.extend([0.0, 0.0, 0.0]),
        }
    }

    // ---- Power (2) ----
    obs.extend([power.current / power.max, power.current]);

    // ---- Mineral surface concentrations under the rover (6) ----
    match chassis_pos {
        Some(pos) => {
            for (_, value) in maps.surface_all_at(pos.x, pos.z) {
                obs.push(value);
            }
        }
        None => obs.extend([0.0; 6]),
    }

    // ---- Beacons remaining (1) ----
    obs.push(reward.beacons_remaining as f32);

    debug_assert_eq!(obs.len(), OBS_DIM, "obs length must match OBS_DIM");
    obs
}
