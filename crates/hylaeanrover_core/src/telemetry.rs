//! RL telemetry: the shared `RoverTelemetry` resource.
//!
//! Sensor systems in `imu.rs` run in Update, compute their values, and
//! write their slice into `RoverTelemetry` as a side effect. The RL
//! observation builder and the headless Python env read it back out.

use bevy::prelude::*;

// ---- Resource: latest telemetry from each sensor -------------------------

#[derive(Resource, Default)]
pub struct RoverTelemetry {
    /// `false` until the rover has spawned and at least one IMU update
    /// has populated the data.
    pub ready: bool,
    pub imu: ImuTelemetry,
    /// 8 ray distances in meters, indexed left → right across the fan.
    /// 200.0 = sensor max range (no hit).
    pub lidar_m: [f32; 8],
    /// Closest-first list of visible cubes (already filtered by cone
    /// + line-of-sight in the cube sensor system).
    pub visible_cubes: Vec<CubeTelemetry>,
    /// Debug-only native sensor state for the nearest cube before the
    /// actionability filter is applied. Not included in the RL observation.
    pub nearest_cube_height_above_ground_m: Option<f32>,
    pub nearest_cube_actionable: bool,
}

#[derive(Default, Clone, Copy)]
pub struct ImuTelemetry {
    pub speed_mps: f32,
    pub heading_deg: f32,
    pub pitch_deg: f32,
    pub roll_deg: f32,
    pub yaw_rate_deg_s: f32,
    pub accel_fwd_m_s2: f32,
    pub accel_lat_m_s2: f32,
}

#[derive(Clone, Copy)]
pub struct CubeTelemetry {
    pub bearing_deg: f32,
    pub distance_m: f32,
}

// ---- Plugin --------------------------------------------------------------

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RoverTelemetry>();
    }
}

// ---- Helpers used by sensor systems --------------------------------------

/// Round to display precision. Sensor systems call these before writing
/// into `RoverTelemetry` so on-screen readouts don't jitter through 6+
/// decimal digits.
pub fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}
pub fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}
