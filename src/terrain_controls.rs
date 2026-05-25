//! Right-side native Bevy UI for live terrain tweaks.
//!
//! Styled like the bevyearth HUD: dark translucent panel with cyan
//! accents. No third-party UI deps — uses Bevy 0.18 UI primitives only.
//!
//! Controls:
//!   - **x2 / x0.5 / +0.1 / -0.1** — step the terrain height scale.
//!   - **Randomize seed** — re-rolls noise + crater placement.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use bevy_rapier3d::prelude::*;

use crate::terrain::{LunarTerrain, LunarTerrainConfig};
use crate::ui::UiFont;

// --- Palette (cribbed from bevyearth so the panel matches the family) ---
const PANEL_BG: Color = Color::srgba(0.03, 0.05, 0.08, 0.82);
const PANEL_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.45);
const TEXT_MAIN: Color = Color::srgba(0.85, 0.95, 1.00, 1.0);
const TEXT_ACCENT: Color = Color::srgba(0.40, 1.00, 1.00, 1.0);
const TEXT_DIM: Color = Color::srgba(0.55, 0.65, 0.75, 1.0);
const BUTTON_BG: Color = Color::srgba(0.05, 0.10, 0.15, 1.0);
const BUTTON_BG_HOVER: Color = Color::srgba(0.10, 0.20, 0.28, 1.0);

const SCALE_MIN: f32 = 0.05;
const SCALE_MAX: f32 = 5.0;

/// Marker on the terrain mesh entity so rebuilds can find and despawn it.
#[derive(Component)]
pub struct TerrainEntity;

/// Marker on the root panel Node. Other systems use this with a
/// `RelativeCursorPosition` query to gate their mouse handling so the
/// 3D camera doesn't react to clicks/drags landed on the panel.
#[derive(Component)]
pub struct TerrainPanel;

#[derive(Resource)]
pub struct TerrainState {
    pub seed: u64,
    pub size: f32,
    pub resolution: usize,
    pub crater_count: usize,
    pub base_heights: Vec<f32>,
    pub height_scale: f32,
    pub last_built_scale: f32,
    pub last_built_seed: u64,
    /// The terrain entity persists across rebuilds; we only mutate its
    /// Mesh asset in place and swap its Collider component. Avoids the
    /// despawn / respawn / Rapier-rebody hitch a full respawn would cause.
    pub terrain_entity: Option<Entity>,
    pub mesh_handle: Option<Handle<Mesh>>,
}

impl TerrainState {
    /// Bilinear height lookup in world space. Multiplies by `height_scale`
    /// so the returned value matches what's actually being rendered /
    /// collided against right now. Returns 0 outside the terrain footprint.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let n = self.resolution;
        if n < 2 {
            return 0.0;
        }
        let u = (x + self.size) / (2.0 * self.size);
        let v = (z + self.size) / (2.0 * self.size);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return 0.0;
        }
        let fx = u * (n - 1) as f32;
        let fz = v * (n - 1) as f32;
        let x0 = (fx.floor() as usize).min(n - 1);
        let z0 = (fz.floor() as usize).min(n - 1);
        let x1 = (x0 + 1).min(n - 1);
        let z1 = (z0 + 1).min(n - 1);
        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;
        let h00 = self.base_heights[z0 * n + x0];
        let h10 = self.base_heights[z0 * n + x1];
        let h01 = self.base_heights[z1 * n + x0];
        let h11 = self.base_heights[z1 * n + x1];
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        (h0 * (1.0 - tz) + h1 * tz) * self.height_scale
    }
}

// --- UI marker components ---
#[derive(Component, Clone, Copy)]
enum ScaleButton {
    DoubleIt,
    HalfIt,
    PlusTenth,
    MinusTenth,
}

#[derive(Component)]
struct ScaleValueText;

#[derive(Component)]
struct SeedValueText;

#[derive(Component)]
struct RandomizeSeedButton;

pub struct TerrainControlsPlugin;

impl Plugin for TerrainControlsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_terrain, setup_ui))
            .add_systems(
                Update,
                (
                    handle_scale_buttons,
                    handle_randomize_button,
                    rebuild_terrain,
                    sync_ui_to_state,
                ),
            );
    }
}

/// True iff the mouse cursor is hovering anywhere inside the terrain
/// panel. Other systems (camera, follow rig) use this to ignore mouse
/// input that would otherwise also rotate the view.
pub fn cursor_over_terrain_panel(
    panel_q: &Query<&RelativeCursorPosition, With<TerrainPanel>>,
) -> bool {
    let Ok(cursor) = panel_q.single() else { return false };
    let Some(p) = cursor.normalized else { return false };
    (0.0..=1.0).contains(&p.x) && (0.0..=1.0).contains(&p.y)
}

// =========================================================================
// Terrain setup / rebuild
// =========================================================================

fn setup_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cfg = LunarTerrainConfig::default();
    let terrain = LunarTerrain::generate(cfg);
    let base_heights = terrain.heights.clone();

    let mut state = TerrainState {
        seed: cfg.seed,
        size: cfg.size,
        resolution: cfg.resolution,
        crater_count: cfg.crater_count,
        base_heights,
        height_scale: 1.0,
        last_built_scale: 1.0,
        last_built_seed: cfg.seed,
        terrain_entity: None,
        mesh_handle: None,
    };

    let (mesh, collider) = build_mesh_and_collider(&state);
    let mesh_handle = meshes.add(mesh);
    let material_handle = materials.add(StandardMaterial {
        // White base_color so per-vertex colors drive the displayed
        // shade. terrain::build_mesh seeds every vertex grey, so the
        // off-state still looks like the old regolith; the minerals
        // module overrides individual vertices to colour-code element
        // concentrations.
        base_color: Color::WHITE,
        perceptual_roughness: 0.95,
        metallic: 0.0,
        reflectance: 0.2,
        ..default()
    });
    let entity = commands
        .spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material_handle),
            RigidBody::Fixed,
            collider,
            // Sandy regolith — bumps the contact friction with the wheels
            // up from the Rapier default of 0.5 so the rover digs in
            // rather than skating across the surface.
            Friction::coefficient(1.5),
            TerrainEntity,
        ))
        .id();
    state.terrain_entity = Some(entity);
    state.mesh_handle = Some(mesh_handle);
    commands.insert_resource(state);
}

fn regenerate_base_heights(state: &mut TerrainState) {
    let cfg = LunarTerrainConfig {
        seed: state.seed,
        size: state.size,
        resolution: state.resolution,
        crater_count: state.crater_count,
    };
    let terrain = LunarTerrain::generate(cfg);
    state.base_heights = terrain.heights;
}

/// Build a fresh visual mesh and heightfield collider for the current
/// state.height_scale. Pure — no commands/entities.
fn build_mesh_and_collider(state: &TerrainState) -> (Mesh, Collider) {
    let n = state.resolution;
    let scale = state.height_scale;
    let scaled_heights: Vec<f32> = state.base_heights.iter().map(|h| h * scale).collect();

    let mut hf_heights = Vec::with_capacity(n * n);
    for x in 0..n {
        for z in 0..n {
            hf_heights.push(scaled_heights[z * n + x]);
        }
    }
    let span = 2.0 * state.size;
    let collider = Collider::heightfield(hf_heights, n, n, Vect::new(span, 1.0, span));

    let mesh = LunarTerrain {
        size: state.size,
        resolution: state.resolution,
        heights: scaled_heights,
    }
    .build_mesh();

    (mesh, collider)
}

const SCALE_EPSILON: f32 = 0.01;

fn rebuild_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut state: ResMut<TerrainState>,
) {
    // Scale only changes on a button press, so dirty-tracking is enough —
    // no need to throttle.
    let scale_changed = (state.height_scale - state.last_built_scale).abs() > SCALE_EPSILON;
    let seed_changed = state.seed != state.last_built_seed;
    if !scale_changed && !seed_changed {
        return;
    }

    if seed_changed {
        regenerate_base_heights(&mut state);
        state.last_built_seed = state.seed;
    }

    let (new_mesh, new_collider) = build_mesh_and_collider(&state);

    // Mutate the existing Mesh asset in place — no asset realloc, no new
    // Handle, GPU upload is a single dirty-data path.
    if let Some(handle) = state.mesh_handle.as_ref() {
        if let Some(mesh) = meshes.get_mut(handle) {
            *mesh = new_mesh;
        }
    }

    // Swap the Collider component on the existing entity rather than
    // despawning + respawning. The RigidBody::Fixed stays put so Rapier
    // doesn't have to re-register the body each rebuild.
    if let Some(entity) = state.terrain_entity {
        commands.entity(entity).insert(new_collider);
    }

    state.last_built_scale = state.height_scale;
}

// =========================================================================
// Native UI
// =========================================================================

fn setup_ui(mut commands: Commands, ui_font: Res<UiFont>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Px(240.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(12.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
            // Lets other systems detect when the cursor is on the panel
            // so they can ignore mouse input that would otherwise also
            // rotate / pan the camera.
            RelativeCursorPosition::default(),
            TerrainPanel,
        ))
        .with_children(|panel| {
            // Title
            panel.spawn((
                Text::new("TERRAIN"),
                ui_font.text(16.0),
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

            // Scale section
            panel
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(6.0),
                    ..default()
                })
                .with_children(|section| {
                    section
                        .spawn(Node {
                            justify_content: JustifyContent::SpaceBetween,
                            flex_direction: FlexDirection::Row,
                            ..default()
                        })
                        .with_children(|row| {
                            row.spawn((
                                Text::new("Height scale"),
                                ui_font.text(12.0),
                                TextColor(TEXT_MAIN),
                            ));
                            row.spawn((
                                Text::new("1.00"),
                                ui_font.text(12.0),
                                TextColor(TEXT_ACCENT),
                                ScaleValueText,
                            ));
                        });

                    // Two rows of two scale buttons each.
                    section
                        .spawn(Node {
                            flex_direction: FlexDirection::Row,
                            column_gap: Val::Px(6.0),
                            ..default()
                        })
                        .with_children(|row| {
                            spawn_scale_button(row, &ui_font, "x2", ScaleButton::DoubleIt);
                            spawn_scale_button(row, &ui_font, "x0.5", ScaleButton::HalfIt);
                        });
                    section
                        .spawn(Node {
                            flex_direction: FlexDirection::Row,
                            column_gap: Val::Px(6.0),
                            ..default()
                        })
                        .with_children(|row| {
                            spawn_scale_button(row, &ui_font, "+0.1", ScaleButton::PlusTenth);
                            spawn_scale_button(row, &ui_font, "-0.1", ScaleButton::MinusTenth);
                        });
                });

            // Randomize Seed button
            spawn_button(panel, &ui_font, "Randomize seed", RandomizeSeedButton);

            // Footer info
            panel.spawn((
                Node {
                    margin: UiRect::top(Val::Auto),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    ..default()
                },
            ))
            .with_children(|footer| {
                footer.spawn((
                    Node {
                        height: Val::Px(1.0),
                        width: Val::Percent(100.0),
                        margin: UiRect::bottom(Val::Px(6.0)),
                        ..default()
                    },
                    BackgroundColor(PANEL_EDGE),
                ));
                footer.spawn((
                    Text::new("seed: --"),
                    ui_font.text(11.0),
                    TextColor(TEXT_DIM),
                    SeedValueText,
                ));
            });
        });
}

fn spawn_button<'a, M: Component>(
    parent: &'a mut ChildSpawnerCommands,
    ui_font: &UiFont,
    label: &str,
    marker: M,
) -> EntityCommands<'a> {
    let mut ec = parent.spawn((
        Button,
        Node {
            width: Val::Percent(100.0),
            height: Val::Px(32.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(BUTTON_BG),
        Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
        marker,
    ));
    ec.with_children(|btn| {
        btn.spawn((
            Text::new(label),
            ui_font.text(12.0),
            TextColor(TEXT_MAIN),
        ));
    });
    ec
}

fn spawn_scale_button(parent: &mut ChildSpawnerCommands, ui_font: &UiFont, label: &str, kind: ScaleButton) {
    parent
        .spawn((
            Button,
            Node {
                flex_grow: 1.0,
                height: Val::Px(28.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BUTTON_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
            kind,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                ui_font.text(13.0),
                TextColor(TEXT_MAIN),
            ));
        });
}

/// Each scale button fires once on the click frame and snaps
/// `height_scale` by its kind. Clamped to [SCALE_MIN, SCALE_MAX].
fn handle_scale_buttons(
    mut q: Query<(&Interaction, &ScaleButton), Changed<Interaction>>,
    mut state: ResMut<TerrainState>,
) {
    for (interaction, kind) in q.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let new_scale = match kind {
            ScaleButton::DoubleIt => state.height_scale * 2.0,
            ScaleButton::HalfIt => state.height_scale * 0.5,
            ScaleButton::PlusTenth => state.height_scale + 0.1,
            ScaleButton::MinusTenth => state.height_scale - 0.1,
        };
        state.height_scale = new_scale.clamp(SCALE_MIN, SCALE_MAX);
    }
}

fn handle_randomize_button(
    mut q: Query<&Interaction, (Changed<Interaction>, With<RandomizeSeedButton>)>,
    mut state: ResMut<TerrainState>,
) {
    for interaction in q.iter_mut() {
        if *interaction == Interaction::Pressed {
            state.seed = rand::random::<u64>();
        }
    }
}

/// Push current state values into the UI text. Also applies hover
/// background colors to buttons.
fn sync_ui_to_state(
    state: Res<TerrainState>,
    mut scale_text: Query<&mut Text, (With<ScaleValueText>, Without<SeedValueText>)>,
    mut seed_text: Query<&mut Text, (With<SeedValueText>, Without<ScaleValueText>)>,
    mut randomize_btn: Query<
        (&Interaction, &mut BackgroundColor),
        (With<RandomizeSeedButton>, Without<ScaleButton>),
    >,
    mut scale_btns: Query<
        (&Interaction, &mut BackgroundColor),
        (With<ScaleButton>, Without<RandomizeSeedButton>),
    >,
) {
    if let Ok(mut t) = scale_text.single_mut() {
        **t = format!("{:.2}", state.height_scale);
    }
    if let Ok(mut t) = seed_text.single_mut() {
        **t = format!("seed: {}", state.seed);
    }

    if let Ok((interaction, mut bg)) = randomize_btn.single_mut() {
        bg.0 = match interaction {
            Interaction::Hovered | Interaction::Pressed => BUTTON_BG_HOVER,
            Interaction::None => BUTTON_BG,
        };
    }

    for (interaction, mut bg) in scale_btns.iter_mut() {
        bg.0 = match interaction {
            Interaction::Hovered | Interaction::Pressed => BUTTON_BG_HOVER,
            Interaction::None => BUTTON_BG,
        };
    }
}
