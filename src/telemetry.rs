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

use crate::minerals::MineralMaps;
use crate::power_cubes::PowerState;
use crate::ui::UiFont;
use crate::ChassisEntity;

// ---- Resource: latest telemetry from each sensor -------------------------

#[derive(Resource, Default)]
pub struct RoverTelemetry {
    /// `false` until the rover has spawned and at least one IMU update
    /// has populated the data.
    pub ready: bool,
    pub imu: ImuTelemetry,
    /// Fixed order: FL, FR, BL, BR.
    pub wheels: [WheelTelemetry; 4],
    /// 8 ray distances in meters, indexed left → right across the fan.
    /// 200.0 = sensor max range (no hit).
    pub lidar_m: [f32; 8],
    /// Closest-first list of visible cubes (already filtered by cone
    /// + line-of-sight in the cube sensor system).
    pub visible_cubes: Vec<CubeTelemetry>,
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

#[derive(Serialize, Default, Clone, Copy)]
pub struct WheelTelemetry {
    pub label: &'static str,
    pub contact: bool,
    pub slip: f32,
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
    wheels: &'a [WheelTelemetry],
    lidar_m: &'a [f32],
    visible_cubes: &'a [CubeTelemetry],
    power: PowerTelemetry,
    minerals_g_m3: Vec<MineralTelemetry>,
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

fn setup_telemetry_ui(mut commands: Commands, ui_font: Res<UiFont>) {
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
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    mut text_q: Query<&mut Text, With<TelemetryText>>,
) {
    let Ok(mut text) = text_q.single_mut() else { return };

    // Sample mineral concentrations at the rover's current position.
    // If the rover hasn't spawned, fall back to the origin so the
    // readout still has plausible numbers instead of being absent.
    let (x, z) = chassis_res
        .0
        .and_then(|e| xforms.get(e).ok())
        .map(|gxf| (gxf.translation().x, gxf.translation().z))
        .unwrap_or((0.0, 0.0));
    let minerals_g_m3 = minerals
        .surface_all_at(x, z)
        .into_iter()
        .map(|(name, surface)| MineralTelemetry { name, surface: round1(surface) })
        .collect();

    let observation = Observation {
        ready: telemetry.ready,
        imu: telemetry.imu,
        wheels: &telemetry.wheels,
        lidar_m: &telemetry.lidar_m,
        visible_cubes: &telemetry.visible_cubes,
        power: PowerTelemetry {
            current_kwh: round3(power.current / 1000.0),
            max_kwh: round3(power.max / 1000.0),
        },
        minerals_g_m3,
    };

    **text = serde_json::to_string(&observation)
        .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
}

// ---- Helpers used by sensor systems --------------------------------------

/// Round to the precision the JSON readout shows. Sensor systems should
/// call these before writing into `RoverTelemetry` so the JSON output
/// doesn't jitter through 6+ decimal digits.
pub fn round1(v: f32) -> f32 { (v * 10.0).round() / 10.0 }
pub fn round2(v: f32) -> f32 { (v * 100.0).round() / 100.0 }
pub fn round3(v: f32) -> f32 { (v * 1000.0).round() / 1000.0 }
