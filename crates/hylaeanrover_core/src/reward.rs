//! Reward function for the rover game / RL environment.
//!
//! Three additive components, all stored on `RewardState`:
//!
//!   - **distance**       — `+1` per meter the chassis has moved.
//!   - **mineral integral** — line integral over the rover's path. Each
//!     frame, for each tracked element, adds `scarcity_weight[i] ·
//!     (surface[i] / base[i]) · |Δposition|` to the running integral.
//!     Integrating over *space* (not time) means a stationary rover
//!     parked on a deposit earns nothing — reward is paid only for
//!     ground newly covered. Dividing by `base[i]` makes all elements
//!     compare on equal footing at baseline (≈ 1.0 per meter at
//!     baseline concentration); the scarcity weights then boost the
//!     rare ones so a rover that finds Ti / H2O / He-3 deposits
//!     scores far more per meter than one crossing Si plains.
//!   - **beacon bonus**   — when the player drops a beacon (capped at
//!     `BEACON_BUDGET`), credits `BEACON_MULTIPLIER · Σ scarcity[i] ·
//!     (subsurface[i] / base[i])` once. Subsurface is the *hidden*
//!     map — beacons reward strategic guessing from surface trends.
//!
//! Top-center HUD shows beacons-remaining and the running total.
//!
//! Resets are triggered by `power_cubes::RelaunchEvent` so a "Relaunch"
//! click clears the reward and refills the beacon budget alongside the
//! rover respawn.

use bevy::prelude::*;

use crate::ChassisEntity;
use crate::minerals::{MineralMaps, element_catalog};
use crate::power_cubes::RelaunchEvent;
use crate::ui::UiFont;

// ---- Tuning constants ----------------------------------------------------

/// Reward per meter of chassis motion.
pub const DISTANCE_REWARD_PER_M: f32 = 1.0;
/// Scalar applied to the mineral line-integral on top of the per-element
/// scarcity weights. Cut from the original 1.0 to keep driving-over-minerals
/// rewarding but make strategic beacon placements clearly worth more.
pub const MINERAL_INTEGRAL_SCALE: f32 = 0.1;
/// Number of beacons the player can place per game.
pub const BEACON_BUDGET: u32 = 5;
/// Multiplier on the beacon's subsurface-weighted score.
pub const BEACON_MULTIPLIER: f32 = 50.0;
/// Per-element scarcity weight, in the same order `element_catalog()`
/// yields. Water + Ti are clearly premium; He-3 the rarest.
pub const SCARCITY_WEIGHTS: [f32; 6] = [
    1.0,  // Si
    2.0,  // Al
    3.0,  // Fe
    10.0, // Ti
    20.0, // H2O
    50.0, // He-3
];

/// Linear-velocity jumps larger than this between frames are treated as
/// teleport (respawn) and skipped to avoid spiking the distance counter.
const TELEPORT_DISTANCE_M: f32 = 5.0;

// ---- Resource ------------------------------------------------------------

#[derive(Resource)]
pub struct RewardState {
    pub distance: f32,
    pub mineral_integral: f32,
    pub beacon_bonus: f32,
    pub beacons_remaining: u32,
    /// Chassis position from the previous frame; `None` on fresh start
    /// or just after a respawn so the next-frame delta isn't a teleport.
    last_chassis_pos: Option<Vec3>,
}

impl Default for RewardState {
    fn default() -> Self {
        Self {
            distance: 0.0,
            mineral_integral: 0.0,
            beacon_bonus: 0.0,
            beacons_remaining: BEACON_BUDGET,
            last_chassis_pos: None,
        }
    }
}

impl RewardState {
    /// Sum of all three components.
    pub fn total(&self) -> f32 {
        self.distance + self.mineral_integral + self.beacon_bonus
    }

    /// Credit the bonus for a beacon dropped at `(x, z)` and decrement
    /// the budget. Returns the amount credited, or `None` if the player
    /// had no beacons left (caller should refuse to place).
    pub fn try_credit_beacon(&mut self, maps: &MineralMaps, x: f32, z: f32) -> Option<f32> {
        if self.beacons_remaining == 0 {
            return None;
        }
        let samples = maps.subsurface_all_at(x, z);
        let weighted: f32 = samples
            .iter()
            .zip(element_catalog())
            .zip(SCARCITY_WEIGHTS.iter())
            .map(
                |(((_name, sub), (_n, base)), w)| {
                    if base > 0.0 { w * (sub / base) } else { 0.0 }
                },
            )
            .sum();
        let bonus = BEACON_MULTIPLIER * weighted;
        self.beacon_bonus += bonus;
        self.beacons_remaining -= 1;
        Some(bonus)
    }
}

// ---- Plugin --------------------------------------------------------------

pub struct RewardPlugin;

impl Plugin for RewardPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RewardState>()
            .add_systems(Startup, setup_reward_ui)
            .add_systems(
                Update,
                (accumulate_path_integral, sync_reward_ui, reset_on_relaunch),
            );
    }
}

// ---- Accumulator ---------------------------------------------------------

/// Single per-frame system that computes the chassis displacement since
/// last frame and credits both reward components against it. Merging
/// distance + mineral integral into one system means we read the
/// chassis position once and avoid sharing `last_chassis_pos` across
/// systems.
///
/// Mineral concentration is sampled at the current frame's position and
/// treated as constant over the step. Per-frame motion is ≪ the
/// mineral field's ~6 m correlation length, so the trapezoid error is
/// negligible.
fn accumulate_path_integral(
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<&GlobalTransform>,
    maps: Res<MineralMaps>,
    mut reward: ResMut<RewardState>,
) {
    let Some(chassis_id) = chassis_res.0 else {
        reward.last_chassis_pos = None;
        return;
    };
    let Ok(gxf) = chassis_q.get(chassis_id) else {
        return;
    };
    let pos = gxf.translation();

    let ds = match reward.last_chassis_pos {
        Some(prev) => {
            let d = (pos - prev).length();
            // The frame after a respawn shows a multi-meter jump from
            // the old location to the new one — drop it so neither
            // counter spikes.
            if d < TELEPORT_DISTANCE_M { d } else { 0.0 }
        }
        None => 0.0,
    };
    reward.last_chassis_pos = Some(pos);

    if ds <= 0.0 {
        return;
    }

    // Distance reward — straight proportional to motion.
    reward.distance += ds * DISTANCE_REWARD_PER_M;

    // Mineral line integral: scarcity-weighted sum of (surface / base),
    // multiplied by the path length the rover just covered.
    let samples = maps.surface_all_at(pos.x, pos.z);
    let mut weighted_concentration = 0.0;
    for ((_, value), (weight, (_, base))) in samples
        .iter()
        .zip(SCARCITY_WEIGHTS.iter().zip(element_catalog()))
    {
        if base > 0.0 {
            weighted_concentration += weight * (value / base);
        }
    }
    reward.mineral_integral += weighted_concentration * ds * MINERAL_INTEGRAL_SCALE;
}

// ---- Reset on relaunch ---------------------------------------------------

fn reset_on_relaunch(mut events: MessageReader<RelaunchEvent>, mut reward: ResMut<RewardState>) {
    if events.read().count() == 0 {
        return;
    }
    *reward = RewardState::default();
}

// ---- UI ------------------------------------------------------------------

const BAR_BG: Color = Color::srgba(0.03, 0.05, 0.08, 0.85);
const BAR_EDGE: Color = Color::srgba(0.10, 0.90, 0.95, 0.45);
const TEXT_ACCENT: Color = Color::srgba(0.40, 1.00, 1.00, 1.0);
const TEXT_DIM: Color = Color::srgba(0.55, 0.65, 0.75, 0.9);

/// Marker for the value Text in each cell of the top-bar. The variant
/// tells `sync_reward_ui` which value to write.
#[derive(Component, Clone, Copy)]
enum RewardCell {
    Beacons,
    Distance,
    Mineral,
    BeaconBonus,
    Total,
}

/// Top-of-screen bar showing beacons-remaining plus the three reward
/// contributions and the grand total — five cells, one per number.
fn setup_reward_ui(mut commands: Commands, ui_font: Option<Res<UiFont>>) {
    let Some(ui_font) = ui_font else { return };
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            // Horizontally centred over the viewport. `left: 50%` puts
            // its leading edge at the centre; the half-width margin
            // shifts it back so the centre of the bar lines up.
            left: Val::Percent(50.0),
            margin: UiRect::left(Val::Px(-320.0)),
            width: Val::Px(640.0),
            padding: UiRect::axes(Val::Px(14.0), Val::Px(6.0)),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .insert((
            BackgroundColor(BAR_BG),
            Outline::new(Val::Px(1.0), Val::Px(0.0), BAR_EDGE),
        ))
        .with_children(|bar| {
            reward_cell(
                bar,
                &ui_font,
                "BEACONS",
                RewardCell::Beacons,
                format!("{}/{}", BEACON_BUDGET, BEACON_BUDGET),
            );
            reward_cell(bar, &ui_font, "DISTANCE", RewardCell::Distance, "0".into());
            reward_cell(bar, &ui_font, "MINERAL", RewardCell::Mineral, "0".into());
            reward_cell(bar, &ui_font, "BEACON", RewardCell::BeaconBonus, "0".into());
            reward_cell(bar, &ui_font, "TOTAL", RewardCell::Total, "0".into());
        });
}

fn reward_cell(
    bar: &mut ChildSpawnerCommands,
    ui_font: &UiFont,
    label: &str,
    kind: RewardCell,
    initial: String,
) {
    bar.spawn(Node {
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        row_gap: Val::Px(2.0),
        ..default()
    })
    .with_children(|cell| {
        cell.spawn((Text::new(label), ui_font.text(10.0), TextColor(TEXT_DIM)));
        cell.spawn((
            Text::new(initial),
            ui_font.text(13.0),
            TextColor(TEXT_ACCENT),
            kind,
        ));
    });
}

fn sync_reward_ui(reward: Res<RewardState>, mut q: Query<(&RewardCell, &mut Text)>) {
    for (cell, mut text) in q.iter_mut() {
        **text = match cell {
            RewardCell::Beacons => format!("{}/{}", reward.beacons_remaining, BEACON_BUDGET),
            RewardCell::Distance => format_total(reward.distance),
            RewardCell::Mineral => format_total(reward.mineral_integral),
            RewardCell::BeaconBonus => format_total(reward.beacon_bonus),
            RewardCell::Total => format_total(reward.total()),
        };
    }
}

/// Thin formatter: pretty-print the running total with thousands
/// separators (' '), no decimals — keeps the bar readable as the
/// number grows from 0 to ~50k.
fn format_total(v: f32) -> String {
    let n = v.round() as i64;
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(' ');
        }
        out.push(c);
    }
    let mut formatted: String = out.chars().rev().collect();
    if n < 0 {
        formatted.insert(0, '-');
    }
    formatted
}

#[allow(dead_code)]
const _: () = {
    // Compile-time check: SCARCITY_WEIGHTS must stay aligned with the
    // element catalog. If the catalog grows or shrinks, fix this array
    // (and the corresponding entries) here too.
    assert!(SCARCITY_WEIGHTS.len() == 6);
};
