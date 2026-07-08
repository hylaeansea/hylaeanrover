//! RL telemetry: shared `RoverTelemetry` resource + a bottom-of-screen
//! JSON readout that shows exactly what an RL agent would receive.
//!
//! The schema is the same shape we'd send over the wire to a Python
//! training loop: snake_case keys, units in field names, simple types.
//!
//! Data flow each frame:
//!   1. Sensor systems in `imu.rs` run in Update, compute their values,
//!      and write their slice into `RoverTelemetry` as a side effect.
//!   2. `format_telemetry_json` runs in PostUpdate, reads the resource
//!      plus `PowerState` and `MineralMaps`, and writes a single-line
//!      JSON string into the bottom Text node.

use bevy::prelude::*;
use serde::Serialize;

use crate::ChassisEntity;
use crate::game_state::{GameOverReason, GameState, GameStatus};
use crate::minerals::MineralMaps;
use crate::power_cubes::PowerState;
use crate::reward::{BEACON_BUDGET, RewardState};
use crate::ui::UiFont;

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

#[derive(Serialize, Default, Clone, Copy)]
pub struct ImuTelemetry {
    pub speed_mps: f32,
    pub heading_deg: f32,
    pub pitch_deg: f32,
    pub roll_deg: f32,
    pub yaw_rate_deg_s: f32,
    pub accel_fwd_m_s2: f32,
    pub accel_lat_m_s2: f32,
}

#[derive(Serialize, Clone, Copy)]
pub struct CubeTelemetry {
    pub bearing_deg: f32,
    pub distance_m: f32,
}

// ---- Wire-format root used only by the serializer ------------------------

#[derive(Serialize)]
struct Observation<'a> {
    ready: bool,
    imu: ImuTelemetry,
    lidar_m: &'a [f32],
    visible_cubes: &'a [CubeTelemetry],
    power: PowerTelemetry,
    minerals_g_m3: Vec<MineralTelemetry>,
    reward: RewardSnapshot,
    beacons: BeaconSnapshot,
    /// `null` while playing, `"out_of_power"` or `"flipped"` once a
    /// terminal condition has been reached.
    game_over: Option<&'static str>,
}

#[derive(Serialize)]
struct PowerTelemetry {
    current_kwh: f32,
    max_kwh: f32,
}

#[derive(Serialize)]
struct MineralTelemetry {
    name: &'static str,
    surface: f32,
}

#[derive(Serialize)]
struct RewardSnapshot {
    total: f32,
    distance: f32,
    mineral_integral: f32,
    beacon_bonus: f32,
}

#[derive(Serialize)]
struct BeaconSnapshot {
    remaining: u32,
    budget: u32,
}

// ---- Plugin --------------------------------------------------------------

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RoverTelemetry>()
            .add_systems(Startup, setup_telemetry_ui)
            // PostUpdate so the JSON sees this frame's sensor writes.
            .add_systems(PostUpdate, format_telemetry_json);
    }
}

// ---- UI ------------------------------------------------------------------

const BAR_BG: Color = Color::srgba(0.02, 0.03, 0.05, 0.88);
const BAR_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.35);
const JSON_TEXT_COLOR: Color = Color::srgba(0.75, 0.95, 1.00, 0.95);

#[derive(Component)]
struct TelemetryText;

fn setup_telemetry_ui(mut commands: Commands, ui_font: Option<Res<UiFont>>) {
    // Headless mode: no UI plugin → no font → skip the panel.
    let Some(ui_font) = ui_font else { return };
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(0.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(BAR_BG),
            // Hairline on top so the bar visually separates from the
            // 3D viewport above it.
            Outline::new(Val::Px(1.0), Val::Px(0.0), BAR_EDGE),
        ))
        .with_children(|bar| {
            bar.spawn((
                Text::new("{\"ready\": false}"),
                ui_font.text(9.0),
                TextColor(JSON_TEXT_COLOR),
                TelemetryText,
            ));
        });
}

// ---- Sync ----------------------------------------------------------------

fn format_telemetry_json(
    telemetry: Res<RoverTelemetry>,
    power: Res<PowerState>,
    minerals: Res<MineralMaps>,
    reward: Res<RewardState>,
    game_state: Res<GameState>,
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    mut text_q: Query<&mut Text, With<TelemetryText>>,
) {
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };

    // Sample mineral concentrations at the rover's current position.
    // If the rover hasn't spawned, fall back to the origin so the
    // readout still has plausible numbers instead of being absent.
    let (x, z) = chassis_res
        .0
        .and_then(|e| xforms.get(e).ok())
        .map(|gxf| (gxf.translation().x, gxf.translation().z))
        .unwrap_or((0.0, 0.0));
    // 4 significant figures handles both Si (~300 000) and He-3 (~0.003)
    // without ever rounding the trace elements to 0.0.
    let minerals_g_m3 = minerals
        .surface_all_at(x, z)
        .into_iter()
        .map(|(name, surface)| MineralTelemetry {
            name,
            surface: round_sig(surface, 4),
        })
        .collect();

    let game_over = match game_state.status {
        GameStatus::Playing => None,
        GameStatus::GameOver(GameOverReason::OutOfPower) => Some("out_of_power"),
        GameStatus::GameOver(GameOverReason::Flipped) => Some("flipped"),
        GameStatus::GameOver(GameOverReason::BeaconsDeployed) => Some("beacons_deployed"),
    };

    let observation = Observation {
        ready: telemetry.ready,
        imu: telemetry.imu,
        lidar_m: &telemetry.lidar_m,
        visible_cubes: &telemetry.visible_cubes,
        power: PowerTelemetry {
            current_kwh: round3(power.current / 1000.0),
            max_kwh: round3(power.max / 1000.0),
        },
        minerals_g_m3,
        reward: RewardSnapshot {
            total: round2(reward.total()),
            distance: round2(reward.distance),
            mineral_integral: round2(reward.mineral_integral),
            beacon_bonus: round2(reward.beacon_bonus),
        },
        beacons: BeaconSnapshot {
            remaining: reward.beacons_remaining,
            budget: BEACON_BUDGET,
        },
        game_over,
    };

    **text =
        serde_json::to_string(&observation).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
}

// ---- Helpers used by sensor systems --------------------------------------

/// Round to the precision the JSON readout shows. Sensor systems should
/// call these before writing into `RoverTelemetry` so the JSON output
/// doesn't jitter through 6+ decimal digits.
pub fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}
pub fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}
pub fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

/// Round to `sig` significant figures. Used for mineral concentrations
/// because they span ~9 orders of magnitude — He-3 is ≈0.003 g/m³ while
/// Si is ≈300 000 g/m³, and a fixed decimal-place round (e.g. round1)
/// floors He-3 to 0.0 forever.
fn round_sig(v: f32, sig: i32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let exp = v.abs().log10().floor() as i32;
    let factor = 10f32.powi(sig - 1 - exp);
    (v * factor).round() / factor
}
