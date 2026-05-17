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

use crate::ChassisEntity;

// ----- Element catalogue --------------------------------------------------

/// Static description of one tracked element. Concentrations are sampled
/// uniformly in `[base − range, base + range]` (clamped non-negative).
struct ElementSpec {
    /// Short label shown on the HUD row.
    name: &'static str,
    /// Mean concentration in g/m³.
    base: f32,
    /// Variation either side of the mean (g/m³).
    range: f32,
}

/// Realistic-ish lunar regolith abundances at 1500 kg/m³ bulk density.
/// Mare basalts and highland soils differ substantially in Ti, Fe, and
/// Al — those are the elements where range > base/3 here. Water is a
/// special case: nearly absent on the surface but locally rich inside
/// polar permanently-shadowed regions, so the noise floor is allowed
/// to hit zero. He-3 is at lunar-trace ppb levels (≈ 9 mg/m³ mean).
const ELEMENTS: &[ElementSpec] = &[
    // Si: ~20 wt% almost everywhere.
    ElementSpec { name: "Si",    base: 300_000.0, range:  20_000.0 },
    // Al: ~7% in mare, ~14% in highlands.
    ElementSpec { name: "Al",    base: 150_000.0, range:  60_000.0 },
    // Fe: ~12% in mare, ~4% in highlands.
    ElementSpec { name: "Fe",    base: 110_000.0, range:  70_000.0 },
    // Ti: 0.4–4.5%, very localised (ilmenite hotspots).
    ElementSpec { name: "Ti",    base:  35_000.0, range:  32_000.0 },
    // H2O: trace globally, up to ~5% in PSRs at the poles.
    ElementSpec { name: "H2O",   base:   8_000.0, range:  25_000.0 },
    // He-3: ~9 mg/m³ mean, slightly higher in Ti-rich regolith.
    ElementSpec { name: "He-3",  base:       0.010, range:      0.006 },
];

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

impl MineralMaps {
    fn generate(seed: u64, size: f32, resolution: usize) -> Self {
        let n = resolution.max(2);
        let mut rng = StdRng::seed_from_u64(seed);

        // Per-element surface map.
        let surface: Vec<Vec<f32>> = ELEMENTS
            .iter()
            .map(|e| layered_value_noise(&mut rng, n, e.base, e.range))
            .collect();

        // Subsurface = surface + small-amplitude extra noise. The extra
        // is scaled to ~70% of the element's surface range, so the
        // hidden deposits are on the same order of magnitude as the
        // surface variation but uncorrelated in position.
        let subsurface: Vec<Vec<f32>> = surface
            .iter()
            .zip(ELEMENTS.iter())
            .map(|(s, e)| {
                let extra = layered_value_noise(&mut rng, n, 0.0, e.range * 0.7);
                s.iter()
                    .zip(extra.iter())
                    .map(|(s, x)| (s + x).max(0.0))
                    .collect()
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

    /// Hidden from the HUD; will drive future beacon-scoring.
    #[allow(dead_code)]
    pub fn subsurface_at(&self, element: usize, x: f32, z: f32) -> f32 {
        self.subsurface
            .get(element)
            .map(|g| lookup(g, self.resolution, self.size, x, z))
            .unwrap_or(0.0)
    }
}

// ----- UI marker components -----------------------------------------------

/// Sits on the readout cell for element `0` (index into `ELEMENTS`).
#[derive(Component)]
struct ElementReadText(usize);

// ----- Plugin -------------------------------------------------------------

pub struct MineralsPlugin;

impl Plugin for MineralsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MineralMaps::generate(MAP_SEED, MAP_SIZE, MAP_RESOLUTION))
            .add_systems(Startup, setup_mineral_ui)
            .add_systems(Update, sync_mineral_ui);
    }
}

// ----- UI -----------------------------------------------------------------

fn setup_mineral_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                // Sit directly under the POWER panel.
                top: Val::Px(110.0),
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
                Text::new("MINERAL SURVEY"),
                TextFont { font_size: 16.0, ..default() },
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
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::SpaceBetween,
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Text::new(e.name),
                            TextFont { font_size: 12.0, ..default() },
                            TextColor(TEXT_MAIN),
                        ));
                        row.spawn((
                            Text::new("-- g/m³"),
                            TextFont { font_size: 12.0, ..default() },
                            TextColor(TEXT_ACCENT),
                            ElementReadText(i),
                        ));
                    });
            }
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

/// Same flavour of multi-octave value noise we use for the terrain,
/// scaled to `[base − range, base + range]` and clamped non-negative.
fn layered_value_noise(rng: &mut StdRng, n: usize, base: f32, range: f32) -> Vec<f32> {
    let mut grid = vec![0.0f32; n * n];
    let layers: [(usize, f32); 4] = [(2, 1.0), (4, 0.5), (8, 0.25), (16, 0.12)];
    let total_amp: f32 = layers.iter().map(|(_, a)| a).sum();

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
        *v = (base + *v * range).max(0.0);
    }
    grid
}
