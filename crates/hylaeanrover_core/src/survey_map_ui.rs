//! Mini-heatmap of the survey coverage grid in the right (terrain) panel.
//!
//! Renders a rover-centered window of the `SurveyCoverage` visit grid
//! into a small `Image` shown through an `ImageNode`. World-axis
//! aligned (image top = world -Z); the rover cell stays highlighted at
//! the center while the window scrolls with it.
//!
//! Headless-safe: `setup` requires `UiFont`, `Assets<Image>`, and the
//! panel slot, none of which exist in the RL build, so both systems
//! no-op there.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::ChassisEntity;
use crate::imu::{PANEL_EDGE, TEXT_DIM, divider, subheader};
use crate::survey_coverage::SurveyCoverage;
use crate::terrain_controls::SurveyMapSlot;
use crate::ui::UiFont;

/// Cells per side of the rover-centered window (600 m at 5 m cells).
pub const MAP_CELLS: usize = 120;
const HALF: i32 = MAP_CELLS as i32 / 2;
/// Redraw cadence; 5 m cells make per-frame GPU uploads pointless.
const REDRAW_SECONDS: f32 = 0.2;

const OUT_OF_BOUNDS: [u8; 4] = [8, 10, 14, 255];
const UNVISITED: [u8; 4] = [16, 24, 32, 255];
const VISITED_LOW: [f32; 3] = [24.0, 90.0, 100.0];
const VISITED_HIGH: [f32; 3] = [235.0, 255.0, 255.0];
/// Visit count at which the visited gradient saturates.
const VISIT_SATURATION: u8 = 4;
const ROVER: [u8; 4] = [255, 190, 80, 255];

#[derive(Resource)]
struct SurveyMapImage {
    handle: Handle<Image>,
    timer: Timer,
}

pub struct SurveyMapUiPlugin;

impl Plugin for SurveyMapUiPlugin {
    fn build(&self, app: &mut App) {
        // PostStartup: the slot is spawned via Startup commands, which
        // have been applied by then — no cross-plugin Startup ordering.
        app.add_systems(PostStartup, setup_survey_map)
            .add_systems(Update, sync_survey_map);
    }
}

fn setup_survey_map(
    mut commands: Commands,
    ui_font: Option<Res<UiFont>>,
    images: Option<ResMut<Assets<Image>>>,
    slot_q: Query<Entity, With<SurveyMapSlot>>,
) {
    let (Some(ui_font), Some(mut images)) = (ui_font, images) else {
        return;
    };
    let Ok(slot) = slot_q.single() else { return };

    let mut image = Image::new_fill(
        Extent3d {
            width: MAP_CELLS as u32,
            height: MAP_CELLS as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &UNVISITED,
        TextureFormat::Rgba8UnormSrgb,
        // MAIN_WORLD keeps the CPU copy so `get_mut` writes work.
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    // One pixel per cell: keep cells crisp instead of blurring.
    image.sampler = ImageSampler::nearest();
    let handle = images.add(image);

    commands.entity(slot).with_children(|slot| {
        divider(slot);
        subheader(slot, &ui_font, "SURVEY MAP");
        slot.spawn((
            Node {
                width: Val::Percent(100.0),
                aspect_ratio: Some(1.0),
                ..default()
            },
            ImageNode::new(handle.clone()),
            Outline::new(Val::Px(1.0), Val::Px(0.0), PANEL_EDGE),
        ));
        slot.spawn((
            Text::new("600 m window · -Z up"),
            ui_font.text(9.0),
            TextColor(TEXT_DIM),
        ));
    });

    commands.insert_resource(SurveyMapImage {
        handle,
        timer: Timer::from_seconds(REDRAW_SECONDS, TimerMode::Repeating),
    });
}

fn sync_survey_map(
    map: Option<ResMut<SurveyMapImage>>,
    images: Option<ResMut<Assets<Image>>>,
    coverage: Res<SurveyCoverage>,
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
    time: Res<Time>,
) {
    let (Some(mut map), Some(mut images)) = (map, images) else {
        return;
    };
    map.timer.tick(time.delta());
    if !map.timer.is_finished() {
        return;
    }

    let (x, z) = chassis_res
        .0
        .and_then(|e| xforms.get(e).ok())
        .map(|gxf| (gxf.translation().x, gxf.translation().z))
        .unwrap_or((0.0, 0.0));
    let Some((center_x, center_z)) = coverage.cell_for_position(Vec2::new(x, z)) else {
        return;
    };

    let Some(image) = images.get_mut(&map.handle) else {
        return;
    };
    let Some(data) = image.data.as_mut() else {
        return;
    };

    for py in 0..MAP_CELLS {
        for px in 0..MAP_CELLS {
            let cx = center_x as i32 + (px as i32 - HALF);
            // Image rows run top-to-bottom while world +Z points "down"
            // the screen in this top view, so row order matches +Z.
            let cz = center_z as i32 + (py as i32 - HALF);
            let visits = (cx >= 0 && cz >= 0)
                .then(|| coverage.visit_at_cell(cx as usize, cz as usize))
                .flatten();
            let rgba = match visits {
                None => OUT_OF_BOUNDS,
                Some(0) => UNVISITED,
                Some(n) => visited_color(n),
            };
            let i = (py * MAP_CELLS + px) * 4;
            data[i..i + 4].copy_from_slice(&rgba);
        }
    }

    // Rover marker: 3x3 block at the window center, drawn last.
    for dz in -1..=1i32 {
        for dx in -1..=1i32 {
            let px = (HALF + dx) as usize;
            let py = (HALF + dz) as usize;
            let i = (py * MAP_CELLS + px) * 4;
            data[i..i + 4].copy_from_slice(&ROVER);
        }
    }
}

fn visited_color(visits: u8) -> [u8; 4] {
    // Anchor a single visit at the dim end so the ramp only spends its
    // range on revisits: 1 -> LOW, then even steps to HIGH at saturation.
    let t = visits.min(VISIT_SATURATION).saturating_sub(1) as f32 / (VISIT_SATURATION - 1) as f32;
    let channel = |low: f32, high: f32| (low + (high - low) * t).round() as u8;
    [
        channel(VISITED_LOW[0], VISITED_HIGH[0]),
        channel(VISITED_LOW[1], VISITED_HIGH[1]),
        channel(VISITED_LOW[2], VISITED_HIGH[2]),
        255,
    ]
}
