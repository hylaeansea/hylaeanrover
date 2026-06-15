//! Playable game binary. Loads `DefaultPlugins` + the core game logic
//! and adds the camera / FPS HUD on top. The headless RL env in
//! `hylaeanrover_py` reuses `RoverCorePlugin` directly with
//! `MinimalPlugins`.

// Bevy ECS systems take many parameters by design — silence the
// too-many-arguments lint crate-wide (also covers the autopilot module).
#![allow(clippy::too_many_arguments)]

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Hdr;
use bevy::transform::TransformSystems;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;

use hylaeanrover_core::terrain_controls::{TerrainPanel, cursor_over_terrain_panel};
use hylaeanrover_core::ui::UiFont;
use hylaeanrover_core::{ChassisEntity, RoverCorePlugin};

mod autopilot;
use autopilot::AutopilotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        // Wireframes around colliders. Disabled by default — toggle with F1.
        .add_plugins(RapierDebugRenderPlugin::default().disabled())
        // FPS / frame-time diagnostics so the on-screen counter has data.
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        // Everything game-logic-related lives in the core crate.
        .add_plugins(RoverCorePlugin::default())
        // Optional ONNX-policy autopilot (no-op unless `--policy` given).
        .add_plugins(AutopilotPlugin)
        .init_resource::<CameraMode>()
        .add_systems(Startup, (setup, setup_fps_text))
        .add_systems(
            Update,
            (update_fps_text, toggle_debug_render, toggle_camera_mode),
        )
        // Run follow-camera AFTER transform propagation so the chassis's
        // GlobalTransform is up to date — otherwise on the frame a freshly
        // respawned chassis appears, its GlobalTransform is still its
        // un-propagated local value (near origin) and the camera lerps a
        // visible jump toward (0,0,0) before snapping back the next frame.
        .add_systems(PostUpdate, follow_camera.after(TransformSystems::Propagate))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: Some(5.0),
            enabled: false,
            ..default()
        },
        FollowCamera::default(),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.0, 0.5, 0.0)),
    ));
}

// ===== FPS HUD ============================================================

#[derive(Component)]
struct FpsText;

fn setup_fps_text(mut commands: Commands, ui_font: Res<UiFont>) {
    commands.spawn((
        Text::new("FPS: --"),
        ui_font.text(18.0),
        TextColor(Color::srgb(1.0, 1.0, 0.4)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            right: Val::Px(12.0),
            ..default()
        },
        FpsText,
    ));
}

fn update_fps_text(diagnostics: Res<DiagnosticsStore>, mut query: Query<&mut Text, With<FpsText>>) {
    let Ok(mut text) = query.single_mut() else {
        return;
    };
    let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };
    let Some(fps) = fps_diag.smoothed() else {
        return;
    };
    **text = format!("FPS: {:>5.1}", fps);
}

fn toggle_debug_render(keyboard: Res<ButtonInput<KeyCode>>, mut ctx: ResMut<DebugRenderContext>) {
    if keyboard.just_pressed(KeyCode::F1) {
        ctx.enabled = !ctx.enabled;
    }
}

// ===== Camera =============================================================

#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    Orbit,
    #[default]
    FollowBehind,
}

#[derive(Component)]
struct FollowCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    sensitivity: f32,
    zoom_sensitivity: f32,
}

impl Default for FollowCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.3,
            distance: 10.0,
            sensitivity: 0.005,
            zoom_sensitivity: 0.1,
        }
    }
}

fn toggle_camera_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<CameraMode>,
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<&GlobalTransform, (Without<Camera3d>, With<RigidBody>)>,
    mut camera_q: Query<&mut PanOrbitCamera>,
) {
    if !keyboard.just_pressed(KeyCode::KeyC) {
        return;
    }
    *mode = match *mode {
        CameraMode::Orbit => CameraMode::FollowBehind,
        CameraMode::FollowBehind => CameraMode::Orbit,
    };
    let Ok(mut pan_orbit) = camera_q.single_mut() else {
        return;
    };
    pan_orbit.enabled = matches!(*mode, CameraMode::Orbit);
    if matches!(*mode, CameraMode::Orbit)
        && let Some(chassis_id) = chassis_res.0
        && let Ok(gxf) = chassis_q.get(chassis_id)
    {
        pan_orbit.target_focus = gxf.translation();
    }
}

fn follow_camera(
    time: Res<Time>,
    mode: Res<CameraMode>,
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<&GlobalTransform, (Without<Camera3d>, With<RigidBody>)>,
    mut camera_q: Query<(&mut Transform, &mut FollowCamera), With<Camera3d>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
    panel_q: Query<&bevy::ui::RelativeCursorPosition, With<TerrainPanel>>,
) {
    if !matches!(*mode, CameraMode::FollowBehind) {
        mouse_motion.read().count();
        scroll.read().count();
        return;
    }

    let Some(chassis_id) = chassis_res.0 else {
        return;
    };
    let Ok(chassis_gxf) = chassis_q.get(chassis_id) else {
        return;
    };
    let chassis_pos = chassis_gxf.translation();
    let chassis_rot = chassis_gxf.rotation();
    let Ok((mut cam_xform, mut follow)) = camera_q.single_mut() else {
        return;
    };

    let over_ui = cursor_over_terrain_panel(&panel_q);

    let mut dx = 0.0_f32;
    let mut dy = 0.0_f32;
    if mouse_buttons.pressed(MouseButton::Left) && !over_ui {
        for ev in mouse_motion.read() {
            dx += ev.delta.x;
            dy += ev.delta.y;
        }
    } else {
        mouse_motion.read().count();
    }
    if dx != 0.0 || dy != 0.0 {
        follow.yaw -= dx * follow.sensitivity;
        follow.pitch = (follow.pitch + dy * follow.sensitivity).clamp(-1.4, 1.4);
    }

    let mut s = 0.0_f32;
    if !over_ui {
        for ev in scroll.read() {
            s += ev.y;
        }
    } else {
        scroll.read().count();
    }
    if s != 0.0 {
        follow.distance = (follow.distance * (1.0 - s * follow.zoom_sensitivity)).clamp(2.0, 100.0);
    }

    let cp = follow.pitch.cos();
    let local_offset = Vec3::new(
        follow.distance * cp * follow.yaw.cos(),
        follow.distance * follow.pitch.sin(),
        follow.distance * cp * follow.yaw.sin(),
    );

    let chassis_fwd = chassis_rot * Vec3::NEG_X;
    let flat = Vec3::new(chassis_fwd.x, 0.0, chassis_fwd.z);
    let yaw_rot = if flat.length_squared() > 1e-4 {
        let flat = flat.normalize();
        Quat::from_rotation_y(flat.z.atan2(-flat.x))
    } else {
        Quat::IDENTITY
    };

    let world_offset = yaw_rot * local_offset;
    let target_pos = chassis_pos + world_offset;

    let alpha = 1.0 - (-time.delta_secs() * 12.0).exp();
    cam_xform.translation = cam_xform.translation.lerp(target_pos, alpha);

    let look_target = chassis_pos + Vec3::Y * 0.5;
    cam_xform.look_at(look_target, Vec3::Y);
}
