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
/// When the rover is coasting (moving but the player isn't pressing W/S),
/// this fraction of the would-be drain is *added* to the battery instead
/// of subtracted — i.e. regenerative braking at 20% efficiency.
const REGEN_EFFICIENCY: f32 = 0.2;
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
const BUTTON_BG: Color = Color::srgba(0.05, 0.10, 0.15, 1.0);
const BUTTON_BG_HOVER: Color = Color::srgba(0.10, 0.20, 0.28, 1.0);

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

#[derive(Component)]
struct TutorialBanner;

#[derive(Component)]
struct GameOverPanel;

#[derive(Component)]
struct RelaunchButton;

/// Banner shown on startup that lists the basic keyboard controls. It
/// fades in, holds for a moment, then fades out and is despawned.
#[derive(Component)]
struct ControlsHint {
    elapsed: f32,
}

/// Tracks the tutorial banner's current rendered alpha so it can fade
/// smoothly toward 0 / 1 each frame.
#[derive(Default)]
struct TutorialFade(f32);

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
    /// How many cubes the player has absorbed so far. Used by the
    /// tutorial banner to dismiss itself after the player has clearly
    /// figured the mechanic out.
    pub pickups_count: u32,
}

impl Default for PowerState {
    fn default() -> Self {
        Self {
            current: POWER_MAX,
            max: POWER_MAX,
            last_chassis_pos: None,
            pickups_count: 0,
        }
    }
}

impl PowerState {
    /// Whether the rover has any energy to spend.
    pub fn has_power(&self) -> bool {
        self.current > 0.0
    }
}

/// Fired by the Relaunch button when the player is out of power and
/// wants to start over. `main` listens for this and re-spawns the rover
/// at the landing pad. Power has already been reset by the time we fire.
/// Bevy 0.18 renamed buffered events to "messages" — same semantics as
/// the old `Event` + `EventReader/Writer`.
#[derive(Message)]
pub struct RelaunchEvent;

// ---- Plugin ---------------------------------------------------------------

pub struct PowerCubesPlugin;

impl Plugin for PowerCubesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PowerCubeAssets>()
            .init_resource::<PoissonSpawner>()
            .init_resource::<PowerState>()
            .add_message::<RelaunchEvent>()
            .add_systems(
                Startup,
                (
                    setup_assets,
                    setup_power_ui,
                    setup_tutorial_ui,
                    setup_game_over_ui,
                    setup_controls_hint,
                ),
            )
            .add_systems(
                Update,
                (
                    spawn_power_cubes,
                    consume_power_from_motion,
                    detect_cube_pickup,
                    advance_charging_cubes,
                    sync_power_ui,
                    update_tutorial_fade,
                    update_controls_hint,
                    update_game_over_visibility,
                    handle_relaunch_button,
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
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    let Some(chassis_id) = chassis_res.0 else {
        power.last_chassis_pos = None;
        return;
    };
    let Ok(chassis_gxf) = xforms.get(chassis_id) else {
        return;
    };
    let pos = chassis_gxf.translation();

    // We're "throttling" if the player is actively asking the motor to
    // spin the wheels and there's energy to do it. If they're coasting
    // (no input, or out of power) but the rover is still moving, we
    // treat the motion as regenerative — sliding wheels can charge the
    // battery at REGEN_EFFICIENCY of the equivalent drive cost.
    let throttling = power.has_power()
        && (keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::KeyS));

    if let Some(last) = power.last_chassis_pos {
        let dist = (pos - last).length();
        if dist.is_finite() && dist > 0.0 {
            let delta_wh = dist * DRAIN_PER_METER;
            if throttling {
                power.current = (power.current - delta_wh).max(0.0);
            } else {
                power.current =
                    (power.current + delta_wh * REGEN_EFFICIENCY).min(power.max);
            }
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
            power.pickups_count = power.pickups_count.saturating_add(1);
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

// ---- Tutorial banner ------------------------------------------------------

fn setup_tutorial_ui(mut commands: Commands) {
    // Sits at the top, anchored just to the right of the POWER panel so
    // the arrow visually points at the bar. Initial alpha = 0 — the
    // fade system below brings it in on hover.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(14.0),
                left: Val::Px(260.0),
                padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                column_gap: Val::Px(0.0),
                ..default()
            },
            BackgroundColor(with_panel_alpha(PANEL_BG, 0.0)),
            Outline::new(Val::Px(1.0), Val::Px(0.0), with_panel_alpha(PANEL_EDGE, 0.0)),
            // Hover detection so the banner stays in once the cursor moves
            // from the POWER panel onto the banner itself.
            RelativeCursorPosition::default(),
            TutorialBanner,
        ))
        .with_children(|banner| {
            banner.spawn((
                Text::new("<- drive over blue cubes to charge"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(with_panel_alpha(TEXT_ACCENT, 0.0)),
            ));
        });
}

/// Fade speed for both the tutorial banner and the controls hint
/// (alpha units per second). 0.25 s for a full 0 → 1 transition.
const FADE_SPEED: f32 = 4.0;

/// Fade the tutorial banner in while the cursor is over the POWER
/// panel, fade it out otherwise. Once the player has picked up at
/// least one cube the banner stays at 0 forever — they've got it.
fn update_tutorial_fade(
    time: Res<Time>,
    power: Res<PowerState>,
    panel_q: Query<&RelativeCursorPosition, With<PowerPanel>>,
    banner_q: Query<&RelativeCursorPosition, (With<TutorialBanner>, Without<PowerPanel>)>,
    tutorial_q: Query<Entity, With<TutorialBanner>>,
    mut bg_q: Query<&mut BackgroundColor>,
    mut text_q: Query<&mut TextColor>,
    mut outline_q: Query<&mut Outline>,
    children_q: Query<&Children>,
    mut state: Local<TutorialFade>,
) {
    // Cursor is "over the hint zone" if it's anywhere inside the POWER
    // panel's rect OR inside the banner's own rect (so the banner doesn't
    // snap away the moment the user moves from the bar onto the banner).
    let inside = |cursor: Option<&RelativeCursorPosition>| {
        cursor
            .and_then(|c| c.normalized)
            .map(|p| (0.0..=1.0).contains(&p.x) && (0.0..=1.0).contains(&p.y))
            .unwrap_or(false)
    };
    let hovering =
        inside(panel_q.single().ok()) || inside(banner_q.single().ok());

    let target = if hovering && power.pickups_count == 0 {
        1.0
    } else {
        0.0
    };

    state.0 = step_toward(state.0, target, FADE_SPEED * time.delta_secs());

    if let Ok(entity) = tutorial_q.single() {
        apply_fade(
            entity,
            state.0,
            &mut bg_q,
            &mut text_q,
            &mut outline_q,
            &children_q,
            PANEL_BG,
            PANEL_EDGE,
            TEXT_ACCENT,
        );
    }
}

// ---- Controls hint --------------------------------------------------------

fn setup_controls_hint(mut commands: Commands) {
    // Bottom-centre, fades in on startup, holds, fades out.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(48.0),
                left: Val::Percent(50.0),
                margin: UiRect::left(Val::Px(-180.0)),
                width: Val::Px(360.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(with_panel_alpha(PANEL_BG, 0.0)),
            Outline::new(Val::Px(1.0), Val::Px(0.0), with_panel_alpha(PANEL_EDGE, 0.0)),
            ControlsHint { elapsed: 0.0 },
        ))
        .with_children(|hint| {
            hint.spawn((
                Text::new("WASD to drive"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(with_panel_alpha(TEXT_MAIN, 0.0)),
            ));
            hint.spawn((
                Text::new("R to reset at current position"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(with_panel_alpha(TEXT_MAIN, 0.0)),
            ));
            hint.spawn((
                Text::new("Shift+R to reset at origin"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(with_panel_alpha(TEXT_MAIN, 0.0)),
            ));
        });
}

/// Timeline:
///   0.0 → 1.0 s  fade in
///   1.0 → 6.0 s  hold visible
///   6.0 → 8.0 s  fade out
///   8.0 +        despawn
const CONTROLS_FADE_IN_END: f32 = 1.0;
const CONTROLS_HOLD_END: f32 = 6.0;
const CONTROLS_FADE_OUT_END: f32 = 8.0;

fn update_controls_hint(
    mut commands: Commands,
    time: Res<Time>,
    mut hint_q: Query<(Entity, &mut ControlsHint)>,
    mut bg_q: Query<&mut BackgroundColor>,
    mut text_q: Query<&mut TextColor>,
    mut outline_q: Query<&mut Outline>,
    children_q: Query<&Children>,
) {
    for (entity, mut hint) in hint_q.iter_mut() {
        hint.elapsed += time.delta_secs();
        let t = hint.elapsed;
        let alpha = if t < CONTROLS_FADE_IN_END {
            t / CONTROLS_FADE_IN_END
        } else if t < CONTROLS_HOLD_END {
            1.0
        } else if t < CONTROLS_FADE_OUT_END {
            1.0 - (t - CONTROLS_HOLD_END) / (CONTROLS_FADE_OUT_END - CONTROLS_HOLD_END)
        } else {
            0.0
        };

        apply_fade(
            entity,
            alpha,
            &mut bg_q,
            &mut text_q,
            &mut outline_q,
            &children_q,
            PANEL_BG,
            PANEL_EDGE,
            TEXT_MAIN,
        );

        if t >= CONTROLS_FADE_OUT_END {
            commands.entity(entity).despawn();
        }
    }
}

// ---- Fade helpers ---------------------------------------------------------

fn step_toward(current: f32, target: f32, max_step: f32) -> f32 {
    let diff = target - current;
    if diff.abs() <= max_step {
        target
    } else {
        current + max_step * diff.signum()
    }
}

/// Returns `base` with its alpha scaled by `mult`.
fn with_panel_alpha(base: Color, mult: f32) -> Color {
    let s = base.to_srgba();
    Color::srgba(s.red, s.green, s.blue, s.alpha * mult.clamp(0.0, 1.0))
}

/// Multiplies the alpha of the panel's background, outline, and all
/// direct-child text by `mult`. The base colours are passed in so we
/// always rescale from the original, not from a previously-faded value.
#[allow(clippy::too_many_arguments)]
fn apply_fade(
    entity: Entity,
    mult: f32,
    bg_q: &mut Query<&mut BackgroundColor>,
    text_q: &mut Query<&mut TextColor>,
    outline_q: &mut Query<&mut Outline>,
    children_q: &Query<&Children>,
    base_bg: Color,
    base_edge: Color,
    base_text: Color,
) {
    if let Ok(mut bg) = bg_q.get_mut(entity) {
        bg.0 = with_panel_alpha(base_bg, mult);
    }
    if let Ok(mut outline) = outline_q.get_mut(entity) {
        outline.color = with_panel_alpha(base_edge, mult);
    }
    if let Ok(children) = children_q.get(entity) {
        for child in children.iter() {
            if let Ok(mut tc) = text_q.get_mut(child) {
                tc.0 = with_panel_alpha(base_text, mult);
            }
        }
    }
}

// ---- Game-over splash + relaunch button -----------------------------------

fn setup_game_over_ui(mut commands: Commands) {
    // Full-screen darkening overlay containing a centred modal card.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            GameOverPanel,
        ))
        .with_children(|root| {
            // Modal card
            root.spawn((
                Node {
                    width: Val::Px(420.0),
                    padding: UiRect::all(Val::Px(28.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(16.0),
                    ..default()
                },
                BackgroundColor(PANEL_BG),
                Outline::new(Val::Px(2.0), Val::Px(0.0), PANEL_EDGE),
            ))
            .with_children(|card| {
                card.spawn((
                    Text::new("OUT OF POWER"),
                    TextFont { font_size: 32.0, ..default() },
                    TextColor(TEXT_ACCENT),
                ));
                card.spawn((
                    Text::new("the rover has run dry."),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(TEXT_MAIN),
                ));
                card.spawn((
                    Text::new("game over"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(TEXT_MAIN),
                ));
                // Relaunch button
                card.spawn((
                    Button,
                    Node {
                        width: Val::Px(200.0),
                        height: Val::Px(40.0),
                        margin: UiRect::top(Val::Px(12.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG),
                    Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
                    RelaunchButton,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Relaunch"),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(TEXT_ACCENT),
                    ));
                });
            });
        });
}

fn update_game_over_visibility(
    power: Res<PowerState>,
    mut panel_q: Query<&mut Node, With<GameOverPanel>>,
    mut button_bg_q: Query<&mut BackgroundColor, With<RelaunchButton>>,
    button_int_q: Query<&Interaction, With<RelaunchButton>>,
) {
    let Ok(mut node) = panel_q.single_mut() else { return };
    let target = if power.current <= 0.0 {
        Display::Flex
    } else {
        Display::None
    };
    if node.display != target {
        node.display = target;
    }

    // Hover tint on the button.
    if let (Ok(mut bg), Ok(interaction)) = (button_bg_q.single_mut(), button_int_q.single()) {
        bg.0 = match interaction {
            Interaction::Hovered | Interaction::Pressed => BUTTON_BG_HOVER,
            Interaction::None => BUTTON_BG,
        };
    }
}

fn handle_relaunch_button(
    mut q: Query<&Interaction, (Changed<Interaction>, With<RelaunchButton>)>,
    mut power: ResMut<PowerState>,
    mut writer: MessageWriter<RelaunchEvent>,
) {
    for interaction in q.iter_mut() {
        if *interaction == Interaction::Pressed {
            power.current = power.max;
            writer.write(RelaunchEvent);
        }
    }
}
