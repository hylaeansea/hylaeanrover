//! Simulated IMU readout panel on the left side of the screen, plus a
//! "visible power cubes" sensor panel directly below it.
//!
//! IMU values (updated every frame from the chassis GlobalTransform/Velocity):
//!   - speed   : |linvel|, in m/s
//!   - heading : compass yaw (CW from world -X), wrapped to [0, 360)
//!   - pitch   : rover-forward axis vs. horizon, deg
//!   - roll    : rover-sideways axis vs. horizon, deg
//!
//! Cube sensor (also updated every frame):
//!   - Iterates `PowerCube` entities, computes bearing + distance relative
//!     to the chassis's forward vector, drops anything outside the ±60°
//!     forward cone, then raycasts to check line-of-sight (rover colliders
//!     excluded via collision groups). The closest N visible cubes are
//!     written into a fixed pool of UI rows; unused rows show "--".
//!
//! All of this is intended as observation features for an RL agent driving
//! the rover.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::power_cubes::PowerCube;
use crate::ui::UiFont;
use crate::{ChassisEntity, ROVER_GROUP};

// Re-use the same neon-on-dark styling the other left-side panels use so
// this slots in visually without a fresh palette.
const PANEL_BG: Color = Color::srgba(0.03, 0.05, 0.08, 0.82);
const PANEL_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.45);
const TEXT_MAIN: Color = Color::srgba(0.85, 0.95, 1.00, 1.0);
const TEXT_ACCENT: Color = Color::srgba(0.40, 1.00, 1.00, 1.0);

/// Marker on the value-cell of each readout row.
#[derive(Component, Clone, Copy)]
enum ImuReadout {
    Speed,
    Heading,
    Pitch,
    Roll,
}

/// Marker on the angle-cell of cube row `i` in the visible-cubes panel.
#[derive(Component)]
struct CubeRowAngle(usize);

/// Marker on the distance-cell of cube row `i`.
#[derive(Component)]
struct CubeRowDist(usize);

/// How wide the rover's "forward view" cone is, each side of centerline.
const VIEW_HALF_ANGLE_DEG: f32 = 60.0;
/// Number of UI rows in the panel — the closest N visible cubes fill them,
/// any extras are dropped, any leftover rows show "--".
const MAX_VISIBLE_CUBES: usize = 6;

pub struct ImuPlugin;

impl Plugin for ImuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_imu_ui, setup_cube_sensor_ui))
            .add_systems(Update, (sync_imu_ui, sync_cube_sensor_ui));
    }
}

fn setup_imu_ui(mut commands: Commands, ui_font: Res<UiFont>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                // Sit directly under the MINERAL SURVEY panel
                // (POWER 0..110, MINERAL ~110..360).
                top: Val::Px(360.0),
                left: Val::Px(0.0),
                width: Val::Px(240.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("IMU / TELEMETRY"),
                ui_font.text(16.0),
                TextColor(TEXT_ACCENT),
            ));
            panel.spawn((
                Node {
                    height: Val::Px(1.0),
                    width: Val::Percent(100.0),
                    margin: UiRect::vertical(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(PANEL_EDGE),
            ));

            row(panel, &ui_font, "speed",   ImuReadout::Speed);
            row(panel, &ui_font, "heading", ImuReadout::Heading);
            row(panel, &ui_font, "pitch",   ImuReadout::Pitch);
            row(panel, &ui_font, "roll",    ImuReadout::Roll);
        });
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

fn sync_imu_ui(
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<(&GlobalTransform, Option<&Velocity>)>,
    mut texts: Query<(&ImuReadout, &mut Text)>,
) {
    // Pull the chassis state once. If the rover hasn't spawned yet (or is
    // mid-respawn), leave the readouts at their placeholder.
    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok((gxf, vel)) = chassis_q.get(chassis_id) else { return };

    let speed_mps = vel.map(|v| v.linvel.length()).unwrap_or(0.0);

    let rot = gxf.rotation();

    // Chassis-forward in world space. The rover model is built so its
    // front points down chassis-local -X (same convention `follow_camera`
    // uses in main.rs when computing the rig yaw).
    let fwd = rot * Vec3::NEG_X;

    // Heading: compass yaw of the forward vector projected onto XZ.
    // Compass convention is degrees *clockwise* from "north" (turning
    // right → heading increases), which is the opposite sign of the
    // math-convention atan2(z, -x). So we negate before wrapping.
    let flat = Vec3::new(fwd.x, 0.0, fwd.z);
    let heading_deg = if flat.length_squared() > 1e-6 {
        let flat = flat.normalize();
        let rad = -flat.z.atan2(-flat.x);
        // Wrap to [0, 360) so the readout doesn't flip sign at due-south.
        rad.to_degrees().rem_euclid(360.0)
    } else {
        0.0
    };

    // Pitch: how far the chassis-forward axis lifts above (or dips below)
    // the horizon. asin(fwd.y) because fwd is unit-length.
    let pitch_deg = fwd.y.clamp(-1.0, 1.0).asin().to_degrees();

    // Roll: rotation about the chassis-forward axis, full [-180°, +180°].
    //
    // asin(chassis_side.y) only gives ±90°, so it can't tell upright from
    // upside-down (both put the sideways axis on the horizon). atan2 of
    // the chassis "left" and "up" vectors' world-Y components recovers
    // the full range. Cross-checking against wheel offsets (main.rs:168):
    // wheel_fl is at +Z and wheel_fr is at -Z, so chassis-local +Z is
    // the rover's *left* side. Up is +Y as usual.
    //   upright       : left.y=0,  up.y=+1 → atan2(0,+1) = 0
    //   right side dn : left.y=+1, up.y=0  → atan2(+1, 0) = +90
    //   upside down   : left.y=0,  up.y=-1 → atan2(0,-1) = +180
    //   left side dn  : left.y=-1, up.y=0  → atan2(-1, 0) = -90
    let chassis_left = rot * Vec3::Z;
    let chassis_up = rot * Vec3::Y;
    let roll_deg = chassis_left.y.atan2(chassis_up.y).to_degrees();

    for (kind, mut text) in texts.iter_mut() {
        **text = match kind {
            ImuReadout::Speed   => format!("{:>5.2} m/s", speed_mps),
            ImuReadout::Heading => format!("{:>5.1}°", heading_deg),
            ImuReadout::Pitch   => format!("{:>+5.1}°", pitch_deg),
            ImuReadout::Roll    => format!("{:>+5.1}°", roll_deg),
        };
    }
}

// ===== Cube sensor panel =================================================

fn setup_cube_sensor_ui(mut commands: Commands, ui_font: Res<UiFont>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                // Just below the IMU panel (IMU sits at 360..~510).
                top: Val::Px(540.0),
                left: Val::Px(0.0),
                width: Val::Px(240.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("VISIBLE CUBES"),
                ui_font.text(16.0),
                TextColor(TEXT_ACCENT),
            ));
            panel.spawn((
                Node {
                    height: Val::Px(1.0),
                    width: Val::Percent(100.0),
                    margin: UiRect::vertical(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(PANEL_EDGE),
            ));

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
                        TextColor(Color::srgba(0.55, 0.65, 0.75, 0.9)),
                    ));
                    r.spawn((
                        Text::new("range"),
                        ui_font.text(10.0),
                        TextColor(Color::srgba(0.55, 0.65, 0.75, 0.9)),
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
) {
    // Collect (entity, bearing_deg, distance_m) for every cube that
    // passes both the cone test and the line-of-sight test, then sort
    // by distance and take the closest MAX_VISIBLE_CUBES.
    let mut visible: Vec<(f32, f32)> = Vec::new();

    if let Some(chassis_id) = chassis_res.0 {
        if let Ok(chassis_gxf) = xforms.get(chassis_id) {
            if let Ok(ctx) = rapier.single() {
                let origin = chassis_gxf.translation();
                let rot = chassis_gxf.rotation();
                // Flatten chassis-forward onto the XZ plane — pitch and
                // roll don't matter for "where the sensor is pointing"
                // and including them would jitter bearings when the
                // rover bobs on its suspension.
                let fwd3 = rot * Vec3::NEG_X;
                let fwd = Vec2::new(fwd3.x, fwd3.z);
                if fwd.length_squared() < 1e-6 {
                    // Chassis is somehow facing straight up/down — nothing
                    // to report this frame.
                } else {
                    let fwd = fwd.normalize();
                    // Ray filter: stay in collision group ALL, but only
                    // collide with everything *except* ROVER_GROUP. This
                    // is the inverse of the rover's own group setup
                    // (main.rs:195) and means our rays pass through the
                    // chassis, wheels, hubs, and knuckles without ever
                    // registering a hit.
                    let groups = CollisionGroups::new(
                        Group::ALL,
                        Group::ALL.difference(ROVER_GROUP),
                    );
                    let filter = QueryFilter::default().groups(groups);

                    for (cube_entity, cube_gxf) in cubes.iter() {
                        let cube_pos = cube_gxf.translation();
                        let delta3 = cube_pos - origin;
                        let delta = Vec2::new(delta3.x, delta3.z);
                        let dist_xz = delta.length();
                        // No range cap — the only filters are the forward
                        // cone and line-of-sight. The epsilon guards
                        // against divide-by-zero if the rover is sitting
                        // exactly on a cube.
                        if dist_xz < 1e-3 {
                            continue;
                        }
                        let to_cube = delta / dist_xz;

                        // Signed bearing: + when cube is on rover's right,
                        // - on the left. cross.y of (fwd × to_cube) gives
                        // the sign, with the convention that matches the
                        // heading readout above.
                        let cross_y = fwd.x * to_cube.y - fwd.y * to_cube.x;
                        let dot = fwd.dot(to_cube);
                        let bearing_deg = cross_y.atan2(dot).to_degrees();

                        if bearing_deg.abs() > VIEW_HALF_ANGLE_DEG {
                            continue;
                        }

                        // Line-of-sight check: shoot a ray from the
                        // chassis straight at the cube center in 3D
                        // (using the actual cube_pos, not the flattened
                        // direction — terrain ridges have height). If
                        // the first hit isn't this cube, something is in
                        // the way.
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

    // Closest first.
    visible.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    visible.truncate(MAX_VISIBLE_CUBES);

    // Pool-style update: fill row i with visible[i] if present, else "--".
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
}
