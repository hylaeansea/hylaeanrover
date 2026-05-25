//! Shared UI plumbing.
//!
//! Bevy's built-in default font is `FiraMono-subset` — ASCII only. Anything
//! outside that range (°, ×, ±, μ, ², arrows…) renders as a blank tofu box.
//! We load DejaVu Sans from `assets/fonts/` at startup and expose it as a
//! `UiFont` resource so every panel can pull the same handle.
//!
//! DejaVu Sans (vs. Fira Sans we used previously) covers the Geometric
//! Shapes (●○) and Block Elements (▁▂▃▄▅▆▇█) ranges that the wheel-contact
//! glyphs and the lidar histogram rely on. Bitstream Vera derivative
//! license — see `assets/fonts/LICENSE.txt`.

use bevy::prelude::*;

/// Cloneable handle to the bundled UI font. Insert as a resource and call
/// `.text(size)` when constructing a `TextFont`.
#[derive(Resource, Clone)]
pub struct UiFont {
    pub regular: Handle<Font>,
}

impl UiFont {
    pub fn text(&self, size: f32) -> TextFont {
        TextFont { font: self.regular.clone(), font_size: size, ..default() }
    }
}

/// Entity of the flex-column container that holds the stacked left-side
/// panels (POWER, MINERAL SURVEY, IMU / TELEMETRY, VISIBLE CUBES). Each
/// panel's setup system reads this resource and inserts `ChildOf(0)` so
/// the panels self-stack without manual `top:` math.
#[derive(Resource, Clone, Copy)]
pub struct LeftSidebar(pub Entity);

/// SystemSet for ordering the panel setup systems. ChildOf-relationships
/// honor insertion order for `Children`, so this guarantees POWER appears
/// at the top and VISIBLE CUBES at the bottom regardless of which plugin
/// registers first.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum LeftSidebarSet {
    Power,
    Mineral,
    Imu,
    Cubes,
}

pub struct UiFontPlugin;

impl Plugin for UiFontPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, (load_ui_fonts, spawn_left_sidebar).chain())
            // Force panel setups into the documented order even though
            // they live in different plugins.
            .configure_sets(
                Startup,
                (
                    LeftSidebarSet::Power,
                    LeftSidebarSet::Mineral,
                    LeftSidebarSet::Imu,
                    LeftSidebarSet::Cubes,
                )
                    .chain(),
            );
    }
}

// Run in PreStartup so the handle exists before any panel's Startup system
// queries the resource.
fn load_ui_fonts(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(UiFont {
        regular: asset_server.load("fonts/DejaVuSans.ttf"),
    });
}

fn spawn_left_sidebar(mut commands: Commands) {
    let entity = commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            // Matches the existing panel width so children fit naturally.
            width: Val::Px(240.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .id();
    commands.insert_resource(LeftSidebar(entity));
}
