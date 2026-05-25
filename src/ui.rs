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

pub struct UiFontPlugin;

impl Plugin for UiFontPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, load_ui_fonts);
    }
}

// Run in PreStartup so the handle exists before any panel's Startup system
// queries the resource.
fn load_ui_fonts(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(UiFont {
        regular: asset_server.load("fonts/DejaVuSans.ttf"),
    });
}
