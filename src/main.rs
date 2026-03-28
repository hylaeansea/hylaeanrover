use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

fn main() {
    App::new()
        // DefaultPlugins gives us a window, renderer, input, asset loading —
        // everything needed to actually see something on screen
        .add_plugins(DefaultPlugins)
        // PanOrbitCameraPlugin adds systems that handle mouse input
        // and update any camera entity that has the PanOrbitCamera component
        .add_plugins(PanOrbitCameraPlugin)
        // Run `setup` once at startup, before the first frame
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    // Commands lets us spawn entities into the world
    mut commands: Commands,
    // AssetServer loads files from the assets/ folder
    asset_server: Res<AssetServer>,
) {
    // --- Camera ---
    // Without a camera, nothing renders. We place it slightly above and behind
    // the origin, pointed at (0,0,0) where the rover will spawn.
    commands.spawn((
        Camera3d::default(),
        // PanOrbitCamera takes over the Transform — we configure it here instead.
        // focus = what point the camera orbits around (the origin, where the rover is)
        // radius = distance from that point (how zoomed in/out)
        // Controls: left-click drag = orbit, right-click drag = pan, scroll = zoom
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: Some(5.0),
            ..default()
        },
    ));

    // --- Light ---
    // A directional light acts like the sun — parallel rays from one direction.
    // We angle it downward (-1.0 on X) and slightly to the side (0.5 on Y)
    // so the rover gets some contrast and casts a shadow.
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.0, 0.5, 0.0)),
    ));

    // --- Rover ---
    // asset_server.load() looks in the assets/ folder. The #Scene0 fragment
    // selects the first scene from the glTF file — this is your entire Blender
    // scene graph (chassis, wheels, etc.) preserved as a hierarchy.
    // SceneRoot spawns all those nodes as children of this entity.
    commands.spawn((
        SceneRoot(asset_server.load("rover_1.glb#Scene0")),
    ));
}
