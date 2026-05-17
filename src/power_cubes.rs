//! Glowing blue power cubes + the rover's energy reserve.
//!
//! - Cubes spawn on a homogeneous Poisson process (interarrival ~ Exp(λ))
//!   inside the current terrain footprint. Each is a dynamic Rapier body
//!   with an HDR-emissive material that picks up the camera's bloom.
//! - The rover keeps an energy reserve (`PowerState`, kWh-scale). Driving
//!   drains it at `DRAIN_PER_METER` Wh per meter of chassis motion.
//! - Bringing the rover next to a cube starts a half-second pickup
//!   animation: the cube's emissive ramps up, then it despawns and
//!   credits `CUBE_VALUE` Wh to the reserve.
//! - The drive system reads `PowerState` and stops the wheels when empty.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use bevy_rapier3d::prelude::*;
use rand::Rng;

use crate::terrain_controls::TerrainState;
use crate::ChassisEntity;

// ---- Tunable constants ----------------------------------------------------
const SPAWN_LAMBDA: f32 = 0.05;
const CUBE_HALF_EXTENT: f32 = 0.35;
const SPAWN_HEIGHT: f32 = 40.0;
/// Maximum |x| / |z| (meters) at which a cube may spawn. Decoupled from
/// terrain size: the arena is now ~5 km, but a cube 2 km away is
/// effectively unreachable on one battery, so we hold spawns inside
/// reasonable round-trip range of the origin.
const SPAWN_EXTENT: f32 = 500.0;

/// Total reserve at startup (Watt-hours). 1 kWh.
const POWER_MAX: f32 = 1000.0;
/// Energy consumed per meter of chassis travel (Wh/m).
const DRAIN_PER_METER: f32 = 0.5;
/// How much energy each cube grants (Wh).
const CUBE_VALUE: f32 = 100.0;
/// Distance (m) from chassis at which a cube starts being absorbed.
const PICKUP_RANGE: f32 = 2.0;
/// Seconds for a cube to fully charge (glow ramp duration).
const CHARGE_TIME: f32 = 0.5;
/// Multiplier on the base emissive value at the peak of the glow ramp.
const PEAK_EMISSIVE_MULT: f32 = 6.0;

// ---- HUD palette (kept in sync with terrain_controls) ---------------------
const PANEL_BG: Color = Color::srgba(0.03, 0.05, 0.08, 0.82);
const PANEL_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.45);
const TEXT_MAIN: Color = Color::srgba(0.85, 0.95, 1.00, 1.0);
const TEXT_ACCENT: Color = Color::srgba(0.40, 1.00, 1.00, 1.0);
const BAR_TRACK: Color = Color::srgba(0.05, 0.08, 0.12, 1.0);
const BAR_FILL: Color = Color::srgba(0.20, 0.85, 0.95, 0.95);

// ---- Components -----------------------------------------------------------

/// Marker on every spawned cube.
#[derive(Component)]
pub struct PowerCube;

/// Added to a cube the moment the rover is close enough to absorb it.
/// While present, the cube's emissive ramps up; when `progress >= 1.0`
/// the cube despawns and credits energy.
#[derive(Component)]
struct Charging {
    progress: f32,
    /// Base emissive captured at the start of absorption so we ramp back
    /// up from the cube's actual idle glow, not a hard-coded constant.
    base_emissive: LinearRgba,
}

#[derive(Component)]
struct PowerBarFill;

#[derive(Component)]
struct PowerText;

/// Marker on the power panel root so other systems can detect cursor
/// over the UI (currently used only as a placeholder — the panel is
/// non-interactive but the marker is cheap to carry).
#[derive(Component)]
pub struct PowerPanel;

// ---- Resources ------------------------------------------------------------

#[derive(Resource, Default)]
struct PowerCubeAssets {
    mesh: Handle<Mesh>,
}

#[derive(Resource)]
struct PoissonSpawner {
    time_to_next: f32,
}

impl Default for PoissonSpawner {
    fn default() -> Self {
        Self { time_to_next: 1.0 }
    }
}

/// The rover's energy reserve in Watt-hours.
#[derive(Resource)]
pub struct PowerState {
    pub current: f32,
    pub max: f32,
    /// Chassis world position last frame; used to compute distance moved.
    last_chassis_pos: Option<Vec3>,
}

impl Default for PowerState {
    fn default() -> Self {
        Self {
            current: POWER_MAX,
            max: POWER_MAX,
            last_chassis_pos: None,
        }
    }
}

impl PowerState {
    /// Whether the rover has any energy to spend.
    pub fn has_power(&self) -> bool {
        self.current > 0.0
    }
}

// ---- Plugin ---------------------------------------------------------------

pub struct PowerCubesPlugin;

impl Plugin for PowerCubesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PowerCubeAssets>()
            .init_resource::<PoissonSpawner>()
            .init_resource::<PowerState>()
            .add_systems(Startup, (setup_assets, setup_power_ui))
            .add_systems(
                Update,
                (
                    spawn_power_cubes,
                    consume_power_from_motion,
                    detect_cube_pickup,
                    advance_charging_cubes,
                    sync_power_ui,
                ),
            );
    }
}

// ---- Asset / UI setup -----------------------------------------------------

fn setup_assets(
    mut assets: ResMut<PowerCubeAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    assets.mesh = meshes.add(Cuboid::new(
        CUBE_HALF_EXTENT * 2.0,
        CUBE_HALF_EXTENT * 2.0,
        CUBE_HALF_EXTENT * 2.0,
    ));
}

fn setup_power_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Px(240.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
            RelativeCursorPosition::default(),
            PowerPanel,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("POWER"),
                TextFont { font_size: 16.0, ..default() },
                TextColor(TEXT_ACCENT),
            ));

            panel.spawn((
                Node {
                    height: Val::Px(1.0),
                    width: Val::Percent(100.0),
                    ..default()
                },
                BackgroundColor(PANEL_EDGE),
            ));

            // Bar track + fill. Fill width % is updated each frame.
            panel
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(16.0),
                        padding: UiRect::all(Val::Px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BAR_TRACK),
                    Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
                ))
                .with_children(|track| {
                    track.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(BAR_FILL),
                        PowerBarFill,
                    ));
                });

            panel.spawn((
                Text::new("1.000 / 1.000 kWh"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(TEXT_MAIN),
                PowerText,
            ));
        });
}

// ---- Spawn cubes (Poisson) ------------------------------------------------

fn spawn_power_cubes(
    mut commands: Commands,
    time: Res<Time>,
    mut spawner: ResMut<PoissonSpawner>,
    assets: Res<PowerCubeAssets>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain: Option<Res<TerrainState>>,
) {
    spawner.time_to_next -= time.delta_secs();
    if spawner.time_to_next > 0.0 {
        return;
    }

    let mut rng = rand::thread_rng();
    let u: f32 = rng.gen_range(f32::EPSILON..1.0_f32);
    spawner.time_to_next = -u.ln() / SPAWN_LAMBDA;

    // Need the terrain to exist so we know the heightfield is ready and
    // so we can spawn cubes a small fixed height *above the local
    // terrain* rather than at a world-space y. Otherwise on the new big
    // crater map they fall ~300 m and hit at terminal velocity.
    let Some(terrain) = terrain.as_ref() else { return };
    let x: f32 = rng.gen_range(-SPAWN_EXTENT..SPAWN_EXTENT);
    let z: f32 = rng.gen_range(-SPAWN_EXTENT..SPAWN_EXTENT);
    let local_y = terrain.height_at(x, z);

    let spin = Vec3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );

    // Per-cube material so that ramping a cube's emissive during pickup
    // only lights that cube, not every cube on the map.
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.05, 0.25, 0.95),
        emissive: LinearRgba::new(0.2, 1.5, 8.0, 1.0),
        perceptual_roughness: 0.35,
        metallic: 0.1,
        ..default()
    });

    commands.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(material),
        Transform::from_xyz(x, local_y + SPAWN_HEIGHT, z),
        RigidBody::Dynamic,
        Collider::cuboid(CUBE_HALF_EXTENT, CUBE_HALF_EXTENT, CUBE_HALF_EXTENT),
        Friction::coefficient(0.5),
        Restitution::coefficient(0.2),
        Velocity {
            linvel: Vec3::ZERO,
            angvel: spin,
        },
        PowerCube,
    ));
}

// ---- Power drain ----------------------------------------------------------

fn consume_power_from_motion(
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    mut power: ResMut<PowerState>,
) {
    let Some(chassis_id) = chassis_res.0 else {
        power.last_chassis_pos = None;
        return;
    };
    let Ok(chassis_gxf) = xforms.get(chassis_id) else {
        return;
    };
    let pos = chassis_gxf.translation();

    if let Some(last) = power.last_chassis_pos {
        let dist = (pos - last).length();
        if dist.is_finite() && dist > 0.0 {
            let drain = dist * DRAIN_PER_METER;
            power.current = (power.current - drain).max(0.0);
        }
    }
    power.last_chassis_pos = Some(pos);
}

// ---- Cube pickup ----------------------------------------------------------

fn detect_cube_pickup(
    mut commands: Commands,
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    cubes: Query<
        (Entity, &GlobalTransform, &MeshMaterial3d<StandardMaterial>),
        (With<PowerCube>, Without<Charging>),
    >,
    materials: Res<Assets<StandardMaterial>>,
) {
    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok(chassis_gxf) = xforms.get(chassis_id) else { return };
    let chassis_pos = chassis_gxf.translation();
    let range_sq = PICKUP_RANGE * PICKUP_RANGE;

    for (entity, cube_gxf, mat_handle) in cubes.iter() {
        let cube_pos = cube_gxf.translation();
        if (chassis_pos - cube_pos).length_squared() <= range_sq {
            let base_emissive = materials
                .get(&mat_handle.0)
                .map(|m| m.emissive)
                .unwrap_or(LinearRgba::BLACK);
            commands.entity(entity).insert(Charging {
                progress: 0.0,
                base_emissive,
            });
        }
    }
}

fn advance_charging_cubes(
    mut commands: Commands,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cubes: Query<(Entity, &mut Charging, &MeshMaterial3d<StandardMaterial>)>,
    mut power: ResMut<PowerState>,
) {
    for (entity, mut charging, mat_handle) in cubes.iter_mut() {
        charging.progress += time.delta_secs() / CHARGE_TIME;
        let t = charging.progress.clamp(0.0, 1.0);

        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let boost = 1.0 + t * (PEAK_EMISSIVE_MULT - 1.0);
            let base = charging.base_emissive;
            mat.emissive = LinearRgba::new(
                base.red * boost,
                base.green * boost,
                base.blue * boost,
                base.alpha,
            );
        }

        if charging.progress >= 1.0 {
            power.current = (power.current + CUBE_VALUE).min(power.max);
            commands.entity(entity).despawn();
        }
    }
}

// ---- UI sync --------------------------------------------------------------

fn sync_power_ui(
    power: Res<PowerState>,
    mut fill_q: Query<&mut Node, With<PowerBarFill>>,
    mut text_q: Query<&mut Text, With<PowerText>>,
) {
    let frac = if power.max > 0.0 {
        (power.current / power.max).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if let Ok(mut n) = fill_q.single_mut() {
        n.width = Val::Percent(frac * 100.0);
    }
    if let Ok(mut t) = text_q.single_mut() {
        let cur_kwh = power.current / 1000.0;
        let max_kwh = power.max / 1000.0;
        **t = format!("{:.3} / {:.3} kWh", cur_kwh, max_kwh);
    }
}
