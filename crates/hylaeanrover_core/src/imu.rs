//! Left-side sensor stack for RL instrumentation.
//!
//! All readouts share one big panel ("IMU / TELEMETRY") with two
//! subsections, plus a separate cubes panel below it.
//!
//! Motion (per-frame, from chassis GlobalTransform + Velocity):
//!   - speed     : |linvel|, m/s
//!   - heading   : compass yaw (CW from world -X), wrapped [0, 360)
//!   - pitch     : rover-forward axis vs. horizon, deg
//!   - roll      : rotation about chassis-forward axis, full [-180°, +180°]
//!   - yaw rate  : world-Y component of chassis angvel, deg/s
//!   - accel fwd : EMA-smoothed longitudinal accel (body frame), m/s²
//!   - accel lat : EMA-smoothed lateral accel (body frame), m/s²
//!
//! Lidar:
//!   - 8 horizontal rays, fanned ±90° around chassis-forward, capped at
//!     200 m. Rendered as a row of 8 Unicode block characters; taller
//!     block = closer hit.
//!
//! Cube sensor (separate panel below the main one):
//!   - Iterates `PowerCube` entities, computes bearing + distance relative
//!     to chassis-forward, drops anything outside ±60° forward cone, then
//!     raycasts to verify line-of-sight (rover excluded via ROVER_GROUP).
//!     Closest N visible cubes fill a fixed pool of UI rows.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::power_cubes::PowerCube;
use crate::telemetry::{round1, round2, CubeTelemetry, ImuTelemetry, RoverTelemetry};
use crate::ui::{LeftSidebar, LeftSidebarSet, UiFont};
use crate::{ChassisEntity, ROVER_GROUP};

// Re-use the same neon-on-dark styling the other left-side panels use so
// this slots in visually without a fresh palette.
const PANEL_BG: Color = Color::srgba(0.03, 0.05, 0.08, 0.82);
const PANEL_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.45);
const TEXT_MAIN: Color = Color::srgba(0.85, 0.95, 1.00, 1.0);
const TEXT_ACCENT: Color = Color::srgba(0.40, 1.00, 1.00, 1.0);
const TEXT_DIM: Color = Color::srgba(0.55, 0.65, 0.75, 0.9);

// ---- Marker components ----------------------------------------------------

#[derive(Component, Clone, Copy)]
enum ImuReadout {
    Speed,
    Heading,
    Pitch,
    Roll,
    YawRate,
    AccelFwd,
    AccelLat,
}

/// The single Text node showing the lidar histogram.
#[derive(Component)]
struct LidarText;

/// Cube row marker for the bearing cell of row `i`.
#[derive(Component)]
struct CubeRowAngle(usize);

/// Cube row marker for the range cell of row `i`.
#[derive(Component)]
struct CubeRowDist(usize);

// ---- Sensor parameters ---------------------------------------------------

const LIDAR_RAYS: usize = 8;
const LIDAR_MAX_RANGE: f32 = 200.0;
const LIDAR_HALF_ANGLE_DEG: f32 = 90.0;

/// Accel low-pass time constant (seconds). Raw frame-to-frame Δv is too
/// noisy to read — 0.1 s smooths it without lagging visibly.
const ACCEL_SMOOTH_TAU: f32 = 0.10;

/// Cube sensor: forward cone half-width.
const VIEW_HALF_ANGLE_DEG: f32 = 60.0;
/// Cube sensor: pool size.
const MAX_VISIBLE_CUBES: usize = 6;

// ---- Plugin --------------------------------------------------------------

pub struct ImuPlugin;

impl Plugin for ImuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_imu_ui.in_set(LeftSidebarSet::Imu))
            .add_systems(Startup, setup_cube_sensor_ui.in_set(LeftSidebarSet::Cubes))
            .add_systems(
                Update,
                (sync_imu_motion, sync_lidar_ui, sync_cube_sensor_ui),
            );
    }
}

/// Shared raycast filter: passes through rover colliders (chassis, wheels,
/// hubs, knuckles) by inverting the rover's collision group setup at
/// main.rs:195. Used by every sensor system in this module.
fn rover_excluded_filter() -> QueryFilter<'static> {
    let groups = CollisionGroups::new(Group::ALL, Group::ALL.difference(ROVER_GROUP));
    QueryFilter::default().groups(groups)
}

// ===== IMU panel setup ====================================================

fn setup_imu_ui(
    mut commands: Commands,
    ui_font: Option<Res<UiFont>>,
    sidebar: Option<Res<LeftSidebar>>,
) {
    // Headless mode: UI plugin not loaded → skip the panel.
    let (Some(ui_font), Some(sidebar)) = (ui_font, sidebar) else { return };
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
            ChildOf(sidebar.0),
        ))
        .with_children(|panel| {
            // --- Header ---
            panel.spawn((
                Text::new("IMU / TELEMETRY"),
                ui_font.text(16.0),
                TextColor(TEXT_ACCENT),
            ));
            divider(panel);

            // --- Motion readouts ---
            row(panel, &ui_font, "speed",     ImuReadout::Speed);
            row(panel, &ui_font, "heading",   ImuReadout::Heading);
            row(panel, &ui_font, "pitch",     ImuReadout::Pitch);
            row(panel, &ui_font, "roll",      ImuReadout::Roll);
            row(panel, &ui_font, "yaw rate",  ImuReadout::YawRate);
            row(panel, &ui_font, "accel fwd", ImuReadout::AccelFwd);
            row(panel, &ui_font, "accel lat", ImuReadout::AccelLat);

            // --- Lidar subsection ---
            subheader(panel, &ui_font, "LIDAR (±90°, 8 rays)");
            panel
                .spawn(Node {
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    ..default()
                })
                .with_children(|r| {
                    r.spawn((
                        Text::new("▁▁▁▁▁▁▁▁"),
                        ui_font.text(14.0),
                        TextColor(TEXT_ACCENT),
                        LidarText,
                    ));
                });
        });
}

fn divider(panel: &mut ChildSpawnerCommands) {
    panel.spawn((
        Node {
            height: Val::Px(1.0),
            width: Val::Percent(100.0),
            margin: UiRect::vertical(Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(PANEL_EDGE),
    ));
}

fn subheader(panel: &mut ChildSpawnerCommands, ui_font: &UiFont, label: &str) {
    panel.spawn((
        Node {
            margin: UiRect::top(Val::Px(4.0)),
            ..default()
        },
        children![(
            Text::new(label),
            ui_font.text(11.0),
            TextColor(TEXT_DIM),
        )],
    ));
    divider(panel);
}

fn row(panel: &mut ChildSpawnerCommands, ui_font: &UiFont, label: &str, kind: ImuReadout) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
            ..default()
        })
        .with_children(|r| {
            r.spawn((
                Text::new(label),
                ui_font.text(12.0),
                TextColor(TEXT_MAIN),
            ));
            r.spawn((
                Text::new("--"),
                ui_font.text(12.0),
                TextColor(TEXT_ACCENT),
                kind,
            ));
        });
}

// ===== Motion + IMU sync ==================================================

fn sync_imu_motion(
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<(&GlobalTransform, Option<&Velocity>)>,
    mut texts: Query<(&ImuReadout, &mut Text)>,
    time: Res<Time>,
    mut prev_linvel: Local<Option<Vec3>>,
    mut accel_smooth: Local<Vec3>,
    mut telemetry: ResMut<RoverTelemetry>,
) {
    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok((gxf, vel)) = chassis_q.get(chassis_id) else { return };

    let linvel = vel.map(|v| v.linvel).unwrap_or(Vec3::ZERO);
    let angvel = vel.map(|v| v.angvel).unwrap_or(Vec3::ZERO);
    let speed_mps = linvel.length();

    let rot = gxf.rotation();

    // Chassis-forward in world space. The rover model is built so its
    // front points down chassis-local -X (same convention `follow_camera`
    // uses in main.rs when computing the rig yaw).
    let fwd = rot * Vec3::NEG_X;

    // Heading: compass yaw of forward, projected onto XZ. Negate to flip
    // from math-convention CCW to compass-convention CW.
    let flat = Vec3::new(fwd.x, 0.0, fwd.z);
    let heading_deg = if flat.length_squared() > 1e-6 {
        let flat = flat.normalize();
        let rad = -flat.z.atan2(-flat.x);
        rad.to_degrees().rem_euclid(360.0)
    } else {
        0.0
    };

    // Pitch: forward axis above/below horizon, ±90°.
    let pitch_deg = fwd.y.clamp(-1.0, 1.0).asin().to_degrees();

    // Roll: full [-180°, +180°] via atan2 of chassis-left vs. chassis-up
    // (chassis-local +Z is the rover's left side per wheel offsets in
    // main.rs:168). Upside-down reads +180°.
    let chassis_left = rot * Vec3::Z;
    let chassis_up = rot * Vec3::Y;
    let roll_deg = chassis_left.y.atan2(chassis_up.y).to_degrees();

    // Yaw rate: world-Y component of angular velocity. For an upright
    // rover this is the rate of compass heading change; while tilted it
    // diverges slightly from "rate about chassis-up" but stays a useful
    // proxy.
    let yaw_rate_deg = angvel.y.to_degrees();

    // Acceleration: finite-diff linvel and project into body frame.
    // Raw values are noisy, so EMA-smooth with a dt-aware alpha so the
    // smoothing constant doesn't shift with framerate.
    let dt = time.delta_secs();
    let world_accel_raw = match *prev_linvel {
        Some(prev) if dt > 1e-6 => (linvel - prev) / dt,
        _ => Vec3::ZERO,
    };
    *prev_linvel = Some(linvel);
    let alpha = 1.0 - (-dt / ACCEL_SMOOTH_TAU).exp();
    *accel_smooth = accel_smooth.lerp(world_accel_raw, alpha);

    // chassis-right in world. +Z is left, so right = -Z in chassis-local.
    let chassis_right = rot * Vec3::NEG_Z;
    let accel_fwd = accel_smooth.dot(fwd);
    let accel_lat = accel_smooth.dot(chassis_right);

    for (kind, mut text) in texts.iter_mut() {
        **text = match kind {
            ImuReadout::Speed    => format!("{:>5.2} m/s",  speed_mps),
            ImuReadout::Heading  => format!("{:>5.1}°",     heading_deg),
            ImuReadout::Pitch    => format!("{:>+5.1}°",    pitch_deg),
            ImuReadout::Roll     => format!("{:>+6.1}°",    roll_deg),
            ImuReadout::YawRate  => format!("{:>+5.1}°/s",  yaw_rate_deg),
            ImuReadout::AccelFwd => format!("{:>+5.2} m/s²", accel_fwd),
            ImuReadout::AccelLat => format!("{:>+5.2} m/s²", accel_lat),
        };
    }

    // Mirror into the telemetry resource for the JSON readout / future
    // RL bridge. Round here so the JSON doesn't show full-precision
    // floats jittering each frame.
    telemetry.imu = ImuTelemetry {
        speed_mps:       round2(speed_mps),
        heading_deg:     round1(heading_deg),
        pitch_deg:       round1(pitch_deg),
        roll_deg:        round1(roll_deg),
        yaw_rate_deg_s:  round1(yaw_rate_deg),
        accel_fwd_m_s2:  round2(accel_fwd),
        accel_lat_m_s2:  round2(accel_lat),
    };
    telemetry.ready = true;
}

// ===== Lidar sync =========================================================

fn sync_lidar_ui(
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    rapier: ReadRapierContext,
    mut text_q: Query<&mut Text, With<LidarText>>,
    mut telemetry: ResMut<RoverTelemetry>,
) {
    // Note the ORDER here matters for headless: the LidarText query
    // returning Err just means "no UI to update" — we must still cast
    // the rays and write the telemetry. Previously this early-returned
    // on missing text and silently left lidar_m at zero for RL.
    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok(gxf) = xforms.get(chassis_id) else { return };
    let Ok(ctx) = rapier.single() else { return };

    // Sensor mast: a half-meter above chassis center so a ray cast on a
    // slight uphill doesn't immediately catch the ground in front.
    let origin = gxf.translation() + Vec3::Y * 0.5;
    let rot = gxf.rotation();
    let fwd3 = rot * Vec3::NEG_X;
    let fwd = Vec3::new(fwd3.x, 0.0, fwd3.z);
    let fwd = if fwd.length_squared() < 1e-6 {
        Vec3::NEG_X
    } else {
        fwd.normalize()
    };

    let filter = rover_excluded_filter();
    let mut chars = String::with_capacity(LIDAR_RAYS);

    for i in 0..LIDAR_RAYS {
        // Even spacing including the endpoints (-90° and +90°).
        let t = i as f32 / (LIDAR_RAYS as f32 - 1.0);
        let angle_deg = -LIDAR_HALF_ANGLE_DEG + 2.0 * LIDAR_HALF_ANGLE_DEG * t;
        // Negative angle = rotate ray toward rover's RIGHT (compass convention).
        // The Y-rotation that turns -X into the desired direction satisfies:
        //   Q(θ) * (-1,0,0) = (-cosθ, 0, sinθ)
        // which puts +sinθ on world +Z. World +Z is rover's LEFT, so positive
        // θ would aim left. Flip sign so positive bearing = ray on the right.
        let rad = -angle_deg.to_radians();
        let dir = Quat::from_rotation_y(rad) * fwd;
        let hit = ctx.cast_ray(origin, dir, LIDAR_MAX_RANGE, true, filter);
        let dist = hit.map(|(_, toi)| toi).unwrap_or(LIDAR_MAX_RANGE);
        chars.push(distance_bucket(dist));
        telemetry.lidar_m[i] = round1(dist);
    }

    // UI update is optional — headless has no LidarText entity, but
    // the telemetry above has already been written.
    if let Ok(mut text) = text_q.single_mut() {
        **text = chars;
    }
}

/// Closer = taller block. ▁ is reserved for "no hit / very far".
fn distance_bucket(d: f32) -> char {
    if d < 3.0   { '█' }
    else if d < 8.0   { '▇' }
    else if d < 15.0  { '▆' }
    else if d < 25.0  { '▅' }
    else if d < 40.0  { '▄' }
    else if d < 60.0  { '▃' }
    else if d < 100.0 { '▂' }
    else              { '▁' }
}

// ===== Cube sensor panel ==================================================

fn setup_cube_sensor_ui(
    mut commands: Commands,
    ui_font: Option<Res<UiFont>>,
    sidebar: Option<Res<LeftSidebar>>,
) {
    let (Some(ui_font), Some(sidebar)) = (ui_font, sidebar) else { return };
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
            ChildOf(sidebar.0),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("VISIBLE CUBES"),
                ui_font.text(16.0),
                TextColor(TEXT_ACCENT),
            ));
            divider(panel);

            // Column headers
            panel
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    ..default()
                })
                .with_children(|r| {
                    r.spawn((
                        Text::new("bearing"),
                        ui_font.text(10.0),
                        TextColor(TEXT_DIM),
                    ));
                    r.spawn((
                        Text::new("range"),
                        ui_font.text(10.0),
                        TextColor(TEXT_DIM),
                    ));
                });

            for i in 0..MAX_VISIBLE_CUBES {
                cube_row(panel, &ui_font, i);
            }
        });
}

fn cube_row(panel: &mut ChildSpawnerCommands, ui_font: &UiFont, i: usize) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
            ..default()
        })
        .with_children(|r| {
            r.spawn((
                Text::new("--"),
                ui_font.text(12.0),
                TextColor(TEXT_MAIN),
                CubeRowAngle(i),
            ));
            r.spawn((
                Text::new("--"),
                ui_font.text(12.0),
                TextColor(TEXT_ACCENT),
                CubeRowDist(i),
            ));
        });
}

fn sync_cube_sensor_ui(
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    cubes: Query<(Entity, &GlobalTransform), With<PowerCube>>,
    rapier: ReadRapierContext,
    mut angle_texts: Query<(&CubeRowAngle, &mut Text), Without<CubeRowDist>>,
    mut dist_texts: Query<(&CubeRowDist, &mut Text), Without<CubeRowAngle>>,
    mut telemetry: ResMut<RoverTelemetry>,
) {
    let mut visible: Vec<(f32, f32)> = Vec::new();

    if let Some(chassis_id) = chassis_res.0 {
        if let Ok(chassis_gxf) = xforms.get(chassis_id) {
            if let Ok(ctx) = rapier.single() {
                let origin = chassis_gxf.translation();
                let rot = chassis_gxf.rotation();
                let fwd3 = rot * Vec3::NEG_X;
                let fwd = Vec2::new(fwd3.x, fwd3.z);
                if fwd.length_squared() >= 1e-6 {
                    let fwd = fwd.normalize();
                    let filter = rover_excluded_filter();

                    for (cube_entity, cube_gxf) in cubes.iter() {
                        let cube_pos = cube_gxf.translation();
                        let delta3 = cube_pos - origin;
                        let delta = Vec2::new(delta3.x, delta3.z);
                        let dist_xz = delta.length();
                        if dist_xz < 1e-3 {
                            continue;
                        }
                        let to_cube = delta / dist_xz;

                        let cross_y = fwd.x * to_cube.y - fwd.y * to_cube.x;
                        let dot = fwd.dot(to_cube);
                        let bearing_deg = cross_y.atan2(dot).to_degrees();

                        if bearing_deg.abs() > VIEW_HALF_ANGLE_DEG {
                            continue;
                        }

                        let dir3 = (cube_pos - origin).normalize();
                        let max_toi = (cube_pos - origin).length() + 0.5;
                        let hit = ctx.cast_ray(origin, dir3, max_toi, true, filter);
                        let los_clear = matches!(hit, Some((e, _)) if e == cube_entity);
                        if !los_clear {
                            continue;
                        }

                        visible.push((bearing_deg, dist_xz));
                    }
                }
            }
        }
    }

    visible.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    visible.truncate(MAX_VISIBLE_CUBES);

    for (row, mut text) in angle_texts.iter_mut() {
        **text = match visible.get(row.0) {
            Some((bearing, _)) => format!("{:>+5.1}°", bearing),
            None => "--".to_string(),
        };
    }
    for (row, mut text) in dist_texts.iter_mut() {
        **text = match visible.get(row.0) {
            Some((_, dist)) => format!("{:>6.1} m", dist),
            None => "--".to_string(),
        };
    }

    // Mirror the closest N into telemetry — the rover's "cube sensor"
    // has finite channel capacity, same N rows as the UI. `visible` was
    // already sorted by distance and truncated to MAX_VISIBLE_CUBES
    // above, so the agent sees the same closest-N the human sees.
    telemetry.visible_cubes = visible
        .into_iter()
        .map(|(bearing_deg, distance_m)| CubeTelemetry {
            bearing_deg: round1(bearing_deg),
            distance_m: round1(distance_m),
        })
        .collect();
}
