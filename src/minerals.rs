//! Mineral survey: procedural concentration maps for several lunar
//! elements + a passive readout of what's *right under the rover*.
//!
//! Two maps are generated per element:
//!   * **Surface** — what the rover's spectrometer reads at its current
//!     location. Shown live on the HUD.
//!   * **Subsurface** — what's actually down in the regolith. Built as
//!     `surface + an extra noise field`, so high subsurface values are
//!     loosely correlated with surface peaks but contain "buried"
//!     deposits the player has to *infer* from surface trends and mark
//!     with beacons. **Subsurface is intentionally hidden from the
//!     HUD** — it exists so future scoring can rate beacon placements.
//!
//! All concentrations are in **grams per cubic meter** of regolith
//! (g/m³). The numeric ranges are sized for ~1500 kg/m³ lunar regolith
//! density, matching published wt% measurements from the Apollo
//! sample analyses (highland vs. mare composition + polar volatiles).

use bevy::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::ui::{LeftSidebar, LeftSidebarSet, UiFont};
use crate::ChassisEntity;

// ----- Element catalogue --------------------------------------------------

/// How the surface concentration field for one element is generated.
#[derive(Clone, Copy)]
enum DepositMode {
    /// Smooth multi-octave value noise. Suitable for major elements that
    /// vary continuously across the regolith.
    Noise,
    /// Sum of `count` anisotropic 2-D Gaussians at random positions,
    /// random sigma_x/sigma_z (in meters) and random rotation. Each
    /// Gaussian's peak height is sampled in [height_min, height_max] ×
    /// element.range. Suitable for ore-style discrete deposits.
    GaussianMixture {
        count: usize,
        sigma_min: f32,
        sigma_max: f32,
        height_min: f32,
        height_max: f32,
    },
}

/// Static description of one tracked element.
struct ElementSpec {
    /// Short label shown on the HUD row.
    name: &'static str,
    /// Mean (background) concentration in g/m³. For `Noise` it's the
    /// noise mean; for `GaussianMixture` it's the floor everywhere
    /// outside any deposit.
    base: f32,
    /// Variation either side of the mean (g/m³) for `Noise`; for GMM
    /// it's the typical peak height of a deposit above the base.
    range: f32,
    mode: DepositMode,
}

/// Realistic-ish lunar regolith abundances at 1500 kg/m³ bulk density.
/// Mare basalts and highland soils differ substantially in Ti, Fe, and
/// Al — those are the elements where range > base/3 here. Ti, H2O and
/// He-3 are modelled as Gaussian-mixture deposits because their real
/// distribution is dominated by a small number of discrete ore bodies
/// (ilmenite-rich hotspots; permanently-shadowed water reservoirs;
/// solar-wind-implanted He-3 enriched patches).
const ELEMENTS: &[ElementSpec] = &[
    ElementSpec {
        // Si: ~20 wt% almost everywhere.
        name: "Si", base: 300_000.0, range: 20_000.0,
        mode: DepositMode::Noise,
    },
    ElementSpec {
        // Al: ~7% in mare, ~14% in highlands.
        name: "Al", base: 150_000.0, range: 60_000.0,
        mode: DepositMode::Noise,
    },
    ElementSpec {
        // Fe: ~12% in mare, ~4% in highlands.
        name: "Fe", base: 110_000.0, range: 70_000.0,
        mode: DepositMode::Noise,
    },
    ElementSpec {
        // Ti: 0.4–4.5% — ilmenite hotspots.
        name: "Ti", base: 8_000.0, range: 60_000.0,
        mode: DepositMode::GaussianMixture {
            count: 28,
            sigma_min: 40.0,
            sigma_max: 220.0,
            height_min: 0.40,
            height_max: 1.10,
        },
    },
    ElementSpec {
        // H2O: trace globally, up to ~5 wt% in PSR-like patches.
        name: "H2O", base: 200.0, range: 50_000.0,
        mode: DepositMode::GaussianMixture {
            count: 14,
            sigma_min: 30.0,
            sigma_max: 140.0,
            height_min: 0.55,
            height_max: 1.40,
        },
    },
    ElementSpec {
        // He-3: tiny trace baseline, slightly enriched in a handful of
        // mature mare-like patches.
        name: "He-3", base: 0.0030, range: 0.020,
        mode: DepositMode::GaussianMixture {
            count: 9,
            sigma_min: 25.0,
            sigma_max: 90.0,
            height_min: 0.45,
            height_max: 1.00,
        },
    },
];

impl ElementSpec {
    /// Concentration value the overlay colour ramp anchors to "grey".
    fn display_min(&self) -> f32 {
        match self.mode {
            DepositMode::Noise => (self.base - self.range).max(0.0),
            // Anything at or near the base is "no deposit" → grey.
            DepositMode::GaussianMixture { .. } => self.base,
        }
    }
    /// Concentration value the overlay colour ramp anchors to full colour.
    fn display_max(&self) -> f32 {
        self.base + self.range
    }
}

/// One distinct color per element, applied to the terrain when that
/// element's overlay is active. Same order/length as `ELEMENTS`.
const ELEMENT_COLORS: &[[f32; 3]] = &[
    [1.00, 0.85, 0.20], // Si  — yellow
    [1.00, 0.55, 0.15], // Al  — orange
    [0.95, 0.30, 0.15], // Fe  — rust red
    [0.85, 0.30, 0.90], // Ti  — magenta (ilmenite)
    [0.20, 0.80, 1.00], // H2O — cyan
    [0.35, 1.00, 0.45], // He-3 — green
];

/// Default per-vertex colour for the terrain when no overlay is active.
/// Matches the grey baseline that `terrain::build_mesh` seeds.
const DEFAULT_TERRAIN_COLOR: [f32; 3] = [0.30, 0.30, 0.30];

// ----- Map generation parameters ------------------------------------------
/// Matches the default terrain half-extent so the mineral grid covers
/// the entire arena.
const MAP_SIZE: f32 = 2475.0;
/// 257² cells — minerals vary smoothly over tens of meters, far coarser
/// than the height field.
const MAP_RESOLUTION: usize = 257;
const MAP_SEED: u64 = 7;

// ----- HUD palette (kept in sync with the other panels) -------------------
const PANEL_BG: Color = Color::srgba(0.03, 0.05, 0.08, 0.82);
const PANEL_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.45);
const TEXT_MAIN: Color = Color::srgba(0.85, 0.95, 1.00, 1.0);
const TEXT_ACCENT: Color = Color::srgba(0.40, 1.00, 1.00, 1.0);

// ----- Resources -----------------------------------------------------------

/// Per-element surface and subsurface grids. Each inner Vec is row-major
/// `[z * n + x]` with length `resolution²`.
#[derive(Resource)]
pub struct MineralMaps {
    pub size: f32,
    pub resolution: usize,
    pub surface: Vec<Vec<f32>>,
    /// Hidden from the player; consumed by future beacon-scoring code.
    #[allow(dead_code)]
    pub subsurface: Vec<Vec<f32>>,
}

/// Which element (if any) the terrain is currently colour-coding for.
/// Index into `ELEMENTS`. `None` = default grey terrain.
#[derive(Resource, Default)]
pub struct MineralOverlay {
    pub element: Option<usize>,
}

impl MineralMaps {
    fn generate(seed: u64, size: f32, resolution: usize) -> Self {
        let n = resolution.max(2);
        let mut rng = StdRng::seed_from_u64(seed);

        // Per-element surface map. Common elements use multi-octave
        // noise (smooth, continuous variation); rare elements use a
        // Gaussian-mixture model (discrete deposits with random size,
        // orientation, and height).
        let surface: Vec<Vec<f32>> = ELEMENTS
            .iter()
            .map(|e| match e.mode {
                DepositMode::Noise => {
                    layered_value_noise(&mut rng, n, e.base, e.range)
                }
                DepositMode::GaussianMixture {
                    count,
                    sigma_min,
                    sigma_max,
                    height_min,
                    height_max,
                } => gaussian_mixture_field(
                    &mut rng, n, size, e.base, e.range, count, sigma_min,
                    sigma_max, height_min, height_max,
                ),
            })
            .collect();

        // Subsurface: surface + extra hidden deposits, generated in the
        // same flavour as the surface so noise elements stay smooth and
        // GMM elements stay spotty. The extra is mean-zero so it
        // perturbs the surface without changing its overall scale.
        let subsurface: Vec<Vec<f32>> = surface
            .iter()
            .zip(ELEMENTS.iter())
            .map(|(s, e)| match e.mode {
                DepositMode::Noise => {
                    let extra = layered_value_noise(&mut rng, n, 0.0, e.range * 0.7);
                    s.iter()
                        .zip(extra.iter())
                        .map(|(s, x)| (s + x).max(0.0))
                        .collect()
                }
                DepositMode::GaussianMixture {
                    count,
                    sigma_min,
                    sigma_max,
                    height_min,
                    height_max,
                } => {
                    // Extra hidden deposits — same flavour, fewer of them.
                    let extra = gaussian_mixture_field(
                        &mut rng,
                        n,
                        size,
                        0.0,
                        e.range,
                        (count / 2).max(1),
                        sigma_min,
                        sigma_max,
                        height_min * 0.6,
                        height_max * 0.8,
                    );
                    s.iter()
                        .zip(extra.iter())
                        .map(|(s, x)| (s + x).max(0.0))
                        .collect()
                }
            })
            .collect();

        Self { size, resolution: n, surface, subsurface }
    }

    pub fn surface_at(&self, element: usize, x: f32, z: f32) -> f32 {
        self.surface
            .get(element)
            .map(|g| lookup(g, self.resolution, self.size, x, z))
            .unwrap_or(0.0)
    }

    /// (name, surface concentration g/m³) for every tracked element at
    /// the given world XZ position. For the telemetry JSON readout.
    pub fn surface_all_at(&self, x: f32, z: f32) -> Vec<(&'static str, f32)> {
        ELEMENTS
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name, self.surface_at(i, x, z)))
            .collect()
    }

    /// Hidden from the HUD; drives beacon-placement scoring in
    /// `reward.rs`.
    pub fn subsurface_at(&self, element: usize, x: f32, z: f32) -> f32 {
        self.subsurface
            .get(element)
            .map(|g| lookup(g, self.resolution, self.size, x, z))
            .unwrap_or(0.0)
    }

    /// (name, subsurface concentration g/m³) for every tracked element
    /// at world XZ. The subsurface map is the *actual* deposit value the
    /// player is trying to infer from surface readings — the reward
    /// system credits beacons placed over high-subsurface spots.
    pub fn subsurface_all_at(&self, x: f32, z: f32) -> Vec<(&'static str, f32)> {
        ELEMENTS
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name, self.subsurface_at(i, x, z)))
            .collect()
    }
}

/// Static catalog: yields `(name, base_concentration_g_m3)` in the same
/// order the sample helpers use. Lets the reward system normalize raw
/// concentrations against baseline without duplicating the per-element
/// constants.
pub fn element_catalog() -> impl Iterator<Item = (&'static str, f32)> + 'static {
    ELEMENTS.iter().map(|e| (e.name, e.base))
}

// ----- UI marker components -----------------------------------------------

/// Sits on the readout cell for element `i` (index into `ELEMENTS`).
#[derive(Component)]
struct ElementReadText(usize);

/// The whole row of element `i` is a Button — click to toggle the
/// terrain overlay for that element.
#[derive(Component)]
struct ElementRowButton(usize);

// ----- Plugin -------------------------------------------------------------

pub struct MineralsPlugin;

impl Plugin for MineralsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MineralMaps::generate(MAP_SEED, MAP_SIZE, MAP_RESOLUTION))
            .init_resource::<MineralOverlay>()
            .add_systems(Startup, setup_mineral_ui.in_set(LeftSidebarSet::Mineral))
            .add_systems(
                Update,
                (
                    sync_mineral_ui,
                    handle_overlay_buttons,
                    sync_row_highlight,
                    apply_mineral_overlay,
                ),
            );
    }
}

// ----- UI -----------------------------------------------------------------

fn setup_mineral_ui(mut commands: Commands, ui_font: Res<UiFont>, sidebar: Res<LeftSidebar>) {
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
                Text::new("MINERAL SURVEY"),
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

            for (i, e) in ELEMENTS.iter().enumerate() {
                panel
                    .spawn((
                        Button,
                        Node {
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::SpaceBetween,
                            padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                            ..default()
                        },
                        // Background is transparent until this row is the
                        // active overlay (sync_row_highlight does that).
                        BackgroundColor(Color::NONE),
                        ElementRowButton(i),
                    ))
                    .with_children(|row| {
                        row.spawn((
                            Text::new(e.name),
                            ui_font.text(12.0),
                            TextColor(TEXT_MAIN),
                        ));
                        row.spawn((
                            Text::new("-- g/m³"),
                            ui_font.text(12.0),
                            TextColor(TEXT_ACCENT),
                            ElementReadText(i),
                        ));
                    });
            }

            // Small hint below the rows.
            panel.spawn((
                Text::new("click a row to colour-code the terrain"),
                ui_font.text(10.0),
                TextColor(Color::srgba(0.55, 0.65, 0.75, 0.9)),
            ));
        });
}

fn sync_mineral_ui(
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    maps: Res<MineralMaps>,
    mut q: Query<(&ElementReadText, &mut Text)>,
) {
    let chassis_pos = chassis_res
        .0
        .and_then(|e| xforms.get(e).ok())
        .map(|g| g.translation());

    for (read, mut text) in q.iter_mut() {
        let v = if let Some(pos) = chassis_pos {
            maps.surface_at(read.0, pos.x, pos.z)
        } else {
            0.0
        };
        **text = format_g_per_m3(v);
    }
}

/// Adaptive g/m³ formatter — big numbers get thousand-commas, small ones
/// get extra precision, trace ones switch to scientific notation so the
/// He-3 row stays readable next to the silicon row.
fn format_g_per_m3(v: f32) -> String {
    if v >= 1_000.0 {
        format!("{} g/m³", format_with_commas(v))
    } else if v >= 1.0 {
        format!("{:.2} g/m³", v)
    } else if v >= 0.01 {
        format!("{:.3} g/m³", v)
    } else {
        format!("{:.2e} g/m³", v)
    }
}

fn format_with_commas(v: f32) -> String {
    let int = v.round() as i64;
    let mut s = int.to_string();
    let mut out = String::new();
    let is_neg = s.starts_with('-');
    if is_neg {
        s.remove(0);
    }
    let bytes = s.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c as char);
    }
    if is_neg {
        format!("-{}", out)
    } else {
        out
    }
}

// ----- Noise / lookup helpers ---------------------------------------------

fn lookup(grid: &[f32], n: usize, size: f32, x: f32, z: f32) -> f32 {
    if n < 2 {
        return 0.0;
    }
    let u = (x + size) / (2.0 * size);
    let v = (z + size) / (2.0 * size);
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
    let h00 = grid[z0 * n + x0];
    let h10 = grid[z0 * n + x1];
    let h01 = grid[z1 * n + x0];
    let h11 = grid[z1 * n + x1];
    let h0 = h00 * (1.0 - tx) + h10 * tx;
    let h1 = h01 * (1.0 - tx) + h11 * tx;
    h0 * (1.0 - tz) + h1 * tz
}

/// Multi-octave value noise scaled to `[base − range, base + range]` and
/// clamped non-negative. The octave weighting is intentionally
/// high-frequency-heavy: dominant features land at grid_size 64
/// (≈ 80 m on the 5 km arena) instead of the 1-2 km regional swells
/// we got from the (2, 4, …) ladder. The denominator (`norm`) is also
/// less than the actual amplitude sum so individual peaks overshoot
/// past ±1 → sharper, "richer" deposits when mapped through `range`.
fn layered_value_noise(rng: &mut StdRng, n: usize, base: f32, range: f32) -> Vec<f32> {
    let mut grid = vec![0.0f32; n * n];
    // (grid_size, amplitude). Peak frequency at grid_size 64.
    let layers: [(usize, f32); 5] = [
        (16, 0.40),
        (32, 0.70),
        (64, 1.00),
        (128, 0.55),
        (256, 0.25),
    ];
    // Sum of amps = 2.90, but normalise by ~60% of that so peaks can
    // briefly hit ≈ 1.7 — concentration spikes become noticeably
    // bigger without changing the per-element base / range tuning.
    let total_amp: f32 = 1.75;

    for (grid_size, amplitude) in layers {
        let stride = grid_size + 1;
        let coarse: Vec<f32> = (0..stride * stride)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();
        for z in 0..n {
            for x in 0..n {
                let fx = (x as f32 / (n - 1) as f32) * grid_size as f32;
                let fz = (z as f32 / (n - 1) as f32) * grid_size as f32;
                let x0 = (fx.floor() as usize).min(grid_size);
                let z0 = (fz.floor() as usize).min(grid_size);
                let x1 = (x0 + 1).min(grid_size);
                let z1 = (z0 + 1).min(grid_size);
                let mut tx = fx - x0 as f32;
                let mut tz = fz - z0 as f32;
                tx = tx * tx * (3.0 - 2.0 * tx);
                tz = tz * tz * (3.0 - 2.0 * tz);
                let v00 = coarse[z0 * stride + x0];
                let v10 = coarse[z0 * stride + x1];
                let v01 = coarse[z1 * stride + x0];
                let v11 = coarse[z1 * stride + x1];
                let v0 = v00 * (1.0 - tx) + v10 * tx;
                let v1 = v01 * (1.0 - tx) + v11 * tx;
                grid[z * n + x] += (v0 * (1.0 - tz) + v1 * tz) * amplitude / total_amp;
            }
        }
    }

    for v in grid.iter_mut() {
        *v = (base + (*v).clamp(-1.0, 1.0) * range).max(0.0);
    }
    grid
}

/// Renders a sum of `count` anisotropic 2-D Gaussians onto an `n×n`
/// grid covering `[-size, +size]²`. Each Gaussian has independent
/// centre, sigma along two orthogonal axes (drawn from
/// `[sigma_min, sigma_max]`), rotation in `[0, 2π)`, and peak height
/// drawn from `[height_min, height_max] × range`. Output is clamped
/// non-negative.
///
/// Heavy lifting is local: each Gaussian only touches cells within
/// roughly 3.5σ of its centre, so total cost is O(count · σ²) and
/// stays fast even at 257² grid resolution.
#[allow(clippy::too_many_arguments)]
fn gaussian_mixture_field(
    rng: &mut StdRng,
    n: usize,
    size: f32,
    base: f32,
    range: f32,
    count: usize,
    sigma_min: f32,
    sigma_max: f32,
    height_min: f32,
    height_max: f32,
) -> Vec<f32> {
    let mut grid = vec![base; n * n];
    let cell_size = 2.0 * size / (n.saturating_sub(1).max(1) as f32);
    let n_i = n as i32;

    for _ in 0..count {
        let cx: f32 = rng.gen_range(-size..size);
        let cz: f32 = rng.gen_range(-size..size);
        let sigma_x: f32 = rng.gen_range(sigma_min..sigma_max);
        let sigma_z: f32 = rng.gen_range(sigma_min..sigma_max);
        let rot: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        let height: f32 = rng.gen_range(height_min..height_max) * range;

        let cos_r = rot.cos();
        let sin_r = rot.sin();
        let two_sx2 = 2.0 * sigma_x * sigma_x;
        let two_sz2 = 2.0 * sigma_z * sigma_z;

        // Bound the iteration to a square around the centre that contains
        // ≥99% of the Gaussian's integral.
        let influence = sigma_x.max(sigma_z) * 3.5;
        let influence_cells = (influence / cell_size).ceil() as i32 + 1;
        let cx_idx = ((cx + size) / cell_size).round() as i32;
        let cz_idx = ((cz + size) / cell_size).round() as i32;
        let x_lo = (cx_idx - influence_cells).max(0) as usize;
        let x_hi = (cx_idx + influence_cells + 1).min(n_i) as usize;
        let z_lo = (cz_idx - influence_cells).max(0) as usize;
        let z_hi = (cz_idx + influence_cells + 1).min(n_i) as usize;

        for z in z_lo..z_hi {
            let wz = (z as f32 / (n - 1) as f32) * 2.0 * size - size;
            let dz = wz - cz;
            for x in x_lo..x_hi {
                let wx = (x as f32 / (n - 1) as f32) * 2.0 * size - size;
                let dx = wx - cx;
                // Rotate (dx, dz) into the deposit's principal-axis frame.
                let lx = cos_r * dx + sin_r * dz;
                let lz = -sin_r * dx + cos_r * dz;
                let g = (-(lx * lx / two_sx2 + lz * lz / two_sz2)).exp();
                grid[z * n + x] += height * g;
            }
        }
    }

    // Clamp non-negative — base is already ≥ 0, so this guards against
    // pathological combinations of `base` and `height`.
    for v in grid.iter_mut() {
        *v = v.max(0.0);
    }
    grid
}

// ----- Overlay buttons + terrain colour-coding ----------------------------

/// Clicking a row toggles that element's overlay. Click the active row
/// again to turn the overlay off entirely.
fn handle_overlay_buttons(
    mut q: Query<(&Interaction, &ElementRowButton), Changed<Interaction>>,
    mut overlay: ResMut<MineralOverlay>,
) {
    for (interaction, row) in q.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        overlay.element = if overlay.element == Some(row.0) {
            None
        } else {
            Some(row.0)
        };
    }
}

/// Tints the active row's background with the element's colour and
/// gives unaffiliated rows a fully transparent background. Also nudges
/// the label colour brighter when active.
fn sync_row_highlight(
    overlay: Res<MineralOverlay>,
    mut rows_q: Query<(
        &ElementRowButton,
        &Interaction,
        &mut BackgroundColor,
        &Children,
    )>,
    mut text_q: Query<&mut TextColor>,
) {
    for (row, interaction, mut bg, children) in rows_q.iter_mut() {
        let is_active = overlay.element == Some(row.0);
        let [r, g, b] = ELEMENT_COLORS[row.0];

        bg.0 = if is_active {
            // Dim, slightly translucent tint so the text stays readable.
            Color::srgba(r * 0.5, g * 0.5, b * 0.5, 0.55)
        } else if *interaction == Interaction::Hovered {
            Color::srgba(r * 0.25, g * 0.25, b * 0.25, 0.35)
        } else {
            Color::NONE
        };

        // Label (first child, the name) colour brightens when active.
        if let Some(first) = children.iter().next() {
            if let Ok(mut tc) = text_q.get_mut(first) {
                tc.0 = if is_active {
                    Color::srgba(r, g, b, 1.0)
                } else {
                    Color::srgba(0.85, 0.95, 1.00, 1.0) // TEXT_MAIN
                };
            }
        }
    }
}

/// Pushes element-coloured per-vertex colours onto the terrain mesh.
/// Runs every frame but only re-bakes when one of the inputs that
/// would change the colours has changed (overlay element, terrain
/// rebuild seed, terrain scale). After a rebuild the system sees a new
/// `last_built_*` value and re-applies the active overlay.
fn apply_mineral_overlay(
    overlay: Res<MineralOverlay>,
    terrain: Res<crate::terrain_controls::TerrainState>,
    maps: Res<MineralMaps>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut last: Local<Option<(Option<usize>, u64, i32)>>,
) {
    let Some(mesh_handle) = terrain.mesh_handle.as_ref() else {
        return;
    };
    // Quantise scale to avoid float-equality flicker.
    let scale_q = (terrain.last_built_scale * 1000.0) as i32;
    let key = (overlay.element, terrain.last_built_seed, scale_q);
    if Some(key) == *last {
        return;
    }

    let Some(mesh) = meshes.get_mut(mesh_handle) else {
        return;
    };

    // Pull positions out so we know each vertex's world XZ.
    let positions: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
        _ => return,
    };

    let colors: Vec<[f32; 4]> = match overlay.element {
        None => vec![[DEFAULT_TERRAIN_COLOR[0], DEFAULT_TERRAIN_COLOR[1], DEFAULT_TERRAIN_COLOR[2], 1.0]; positions.len()],
        Some(i) => {
            let spec = &ELEMENTS[i];
            // Anchor the colour ramp's "grey" and "full colour" ends to
            // the per-element display range. For Noise elements this is
            // the symmetric `[base − range, base + range]`; for GMM
            // elements `display_min = base`, so the deposit-free
            // background is fully grey.
            let low = spec.display_min();
            let high = spec.display_max();
            let span = (high - low).max(f32::EPSILON);
            let el = ELEMENT_COLORS[i];
            positions
                .iter()
                .map(|p| {
                    let c = maps.surface_at(i, p[0], p[2]);
                    let t = ((c - low) / span).clamp(0.0, 1.0);
                    [
                        DEFAULT_TERRAIN_COLOR[0] * (1.0 - t) + el[0] * t,
                        DEFAULT_TERRAIN_COLOR[1] * (1.0 - t) + el[1] * t,
                        DEFAULT_TERRAIN_COLOR[2] * (1.0 - t) + el[2] * t,
                        1.0,
                    ]
                })
                .collect()
        }
    };

    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    *last = Some(key);
}
