use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Hdr;
use bevy::transform::TransformSystems;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;

mod beacons;
mod imu;
mod minerals;
mod power_cubes;
mod terrain;
mod terrain_controls;
mod ui;
use beacons::BeaconsPlugin;
use imu::ImuPlugin;
use minerals::MineralsPlugin;
use power_cubes::PowerCubesPlugin;
use ui::{UiFont, UiFontPlugin};
use terrain_controls::{cursor_over_terrain_panel, TerrainControlsPlugin, TerrainPanel};

fn main() {
    App::new()
        // DefaultPlugins gives us a window, renderer, input, asset loading —
        // everything needed to actually see something on screen
        .add_plugins(DefaultPlugins)
        // Loads bundled Fira Sans into a `UiFont` resource — every panel
        // pulls its TextFont from here so non-ASCII glyphs render.
        .add_plugins(UiFontPlugin)
        // PanOrbitCameraPlugin adds systems that handle mouse input
        // and update any camera entity that has the PanOrbitCamera component
        .add_plugins(PanOrbitCameraPlugin)
        // Rapier physics engine — simulates gravity, collisions, joints
        // NoUserData means we don't attach custom data to colliders
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        // Wireframes around colliders. Disabled by default because rendering
        // every heightfield triangle edge as a gizmo line is very slow on a
        // 161×161 terrain. Toggle with F1.
        .add_plugins(RapierDebugRenderPlugin::default().disabled())
        // FPS / frame-time diagnostics so the on-screen counter has data
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        // Owns the terrain entity lifecycle, the right-side egui panel,
        // and the rebuild logic when the user tweaks scale or seed.
        .add_plugins(TerrainControlsPlugin)
        // Glowing blue cubes that spawn on a Poisson process across the map.
        .add_plugins(PowerCubesPlugin)
        // Press B to drop a survey beacon glTF behind the rover.
        .add_plugins(BeaconsPlugin)
        // Surface + subsurface mineral maps, drive-by sampler, HUD readout.
        .add_plugins(MineralsPlugin)
        // Left-side HUD with speed / heading / pitch / roll — IMU-like
        // telemetry for RL observation features.
        .add_plugins(ImuPlugin)
        // Initialize the chassis resource as empty — attach_physics will fill it
        .init_resource::<ChassisEntity>()
        .init_resource::<CameraMode>()
        // Run `setup` once at startup, before the first frame
        .add_systems(Startup, (setup, setup_fps_text))
        // Two-phase physics setup:
        // 1. attach_colliders finds glTF entities by name and adds colliders
        // 2. attach_joints runs after, connecting wheels to chassis with hinges
        .add_systems(Update, (attach_colliders, attach_joints).chain())
        .add_systems(Update, drive)
        .add_systems(Update, respawn_rover)
        .add_systems(Update, spawn_initial_rover)
        .add_systems(Update, handle_relaunch_event)
        .add_systems(Update, (update_fps_text, toggle_debug_render, toggle_camera_mode))
        // Run follow-camera AFTER transform propagation so the chassis's
        // GlobalTransform is up to date — otherwise on the frame a freshly
        // respawned chassis appears, its GlobalTransform is still its
        // un-propagated local value (near origin) and the camera lerps a
        // visible jump toward (0,0,0) before snapping back the next frame.
        .add_systems(
            PostUpdate,
            follow_camera.after(TransformSystems::Propagate),
        )
        .run();
}

fn setup(
    // Commands lets us spawn entities into the world
    mut commands: Commands,
) {
    // --- Camera ---
    // Without a camera, nothing renders. We place it slightly above and behind
    // the origin, pointed at (0,0,0) where the rover will spawn.
    commands.spawn((
        Camera3d::default(),
        // HDR is its own marker component in Bevy 0.18. Required for the
        // bloom post-process pass — without it, emissive values > 1.0 in
        // linear RGB get clipped instead of glowing.
        Hdr,
        // Tonemapping maps the HDR output back to displayable LDR.
        Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        // PanOrbitCamera takes over the Transform — we configure it here instead.
        // focus = what point the camera orbits around (the origin, where the rover is)
        // radius = distance from that point (how zoomed in/out)
        // Controls: left-click drag = orbit, right-click drag = pan, scroll = zoom
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: Some(5.0),
            // Start disabled because the default camera mode is FollowBehind
            // — toggle_camera_mode flips this on when the user presses C.
            enabled: false,
            ..default()
        },
        // Follow-rig state used when CameraMode::FollowBehind is active.
        // yaw/pitch/distance are in the chassis's local frame so the rig
        // keeps its pose relative to the rover as the rover turns.
        FollowCamera::default(),
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

    // Terrain is spawned by TerrainControlsPlugin's startup system.

    // The rover is spawned by `spawn_initial_rover` once the terrain
    // resource is available — that way we can place it just above the
    // landing pad (which is at the bottom of the giant arena crater)
    // instead of dropping it from y = 1.5 in world space and watching
    // it free-fall 300 m.
}

/// Clearance above the local terrain at which the rover is spawned. The
/// rover then settles onto the surface under gravity.
const ROVER_SPAWN_CLEARANCE: f32 = 1.5;

/// Spawns the rover the first frame the `TerrainState` resource is
/// available, placing it `ROVER_SPAWN_CLEARANCE` above whatever the
/// terrain height is at the origin (i.e., the landing pad).
fn spawn_initial_rover(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    terrain: Option<Res<terrain_controls::TerrainState>>,
    existing: Query<(), With<RoverRoot>>,
) {
    if !existing.is_empty() {
        return;
    }
    let Some(terrain) = terrain else { return };
    let pad_height = terrain.height_at(0.0, 0.0);
    commands.spawn((
        RoverRoot,
        SceneRoot(asset_server.load("rover_1.glb#Scene0")),
        Transform::from_xyz(0.0, pad_height + ROVER_SPAWN_CLEARANCE, 0.0),
    ));
}

/// Marker component on the rover root entity so we can find it to despawn.
#[derive(Component)]
pub struct RoverRoot;

/// Stores the chassis Entity so wheels can create joints back to it,
/// the follow camera can target it, and the power system can read its
/// position. `pub` so submodules can `use crate::ChassisEntity`.
#[derive(Resource, Default)]
pub struct ChassisEntity(pub Option<Entity>);

/// Wheel positions relative to the chassis origin, in Bevy coordinates.
/// Derived from Blender scene data via MCP:
///   Blender chassis is at (0, 0, 0.66), wheels at Z=0
///   So wheels sit 0.66 below chassis center in Bevy Y
const WHEEL_OFFSETS: &[(&str, Vec3)] = &[
    ("wheel_fl", Vec3::new(-2.49, -0.66, 1.78)),
    ("wheel_fr", Vec3::new(-2.49, -0.66, -1.73)),
    ("wheel_bl", Vec3::new(2.10, -0.66, 1.78)),
    ("wheel_br", Vec3::new(2.10, -0.66, -1.73)),
];

/// Runs every frame. Waits for the glTF scene children to appear, then
/// attaches physics colliders and joints based on their Blender names.
/// Without<Collider> ensures we only process each entity once.
/// Attaches colliders to glTF entities once they appear.
/// Without<Collider> ensures we only process each entity once.
fn attach_colliders(
    mut commands: Commands,
    query: Query<(Entity, &Name), Without<Collider>>,
    mut chassis_res: ResMut<ChassisEntity>,
) {
    // Rover parts (chassis + wheels) sit in their own collision group
    // whose filter excludes itself. Why: the wheel cylinder colliders
    // overlap the chassis cuboid by ~1 m in X and Y at their joints,
    // and Rapier doesn't auto-disable contacts between
    // chain-joint-connected bodies. Without this, every frame Rapier
    // generates a separation impulse between wheel and chassis that
    // fights the suspension/steering joints — visible as wheel jitter.
    // Putting them in a self-excluding group means rover-internal
    // colliders simply don't pair, while terrain and cubes (default
    // group = ALL, filter = ALL) still collide with them normally.
    let rover_groups = CollisionGroups::new(ROVER_GROUP, Group::ALL.difference(ROVER_GROUP));

    for (entity, name) in query.iter() {
        match name.as_str() {
            "chassis" => {
                commands.entity(entity).insert((
                    RigidBody::Dynamic,
                    Collider::compound(vec![(
                        Vect::new(0.0, 0.75, 0.0),
                        Quat::IDENTITY,
                        Collider::cuboid(2.775, 1.5, 1.0),
                    )]),
                    // Rapier puts stationary bodies to sleep to save CPU,
                    // but a sleeping wheel ignores motor velocity changes
                    // until something jolts it awake — which makes W/A/S/D
                    // appear dead when the rover is fully at rest. Disable.
                    Sleeping::disabled(),
                    // Rapier only writes linvel/angvel back to entities
                    // that explicitly carry a Velocity component. The
                    // steering-taper code and the IMU readout both read it.
                    Velocity::default(),
                    rover_groups,
                ));
                chassis_res.0 = Some(entity);
            }
            "wheel_fl" | "wheel_fr" | "wheel_bl" | "wheel_br" => {
                let wheel_rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
                commands.entity(entity).insert((
                    RigidBody::Dynamic,
                    Collider::compound(vec![(
                        Vect::ZERO,
                        wheel_rotation,
                        Collider::cylinder(0.68, 1.12),
                    )]),
                    // High friction for knobby off-road tires. Combined
                    // with the terrain's 1.5 friction (Rapier averages
                    // the two), the effective μ ≈ 2.0 — feels like
                    // driving through compacted sand, not skating on ice.
                    Friction::coefficient(2.5),
                    Sleeping::disabled(),
                    rover_groups,
                    Wheel,
                ));
            }
            _ => {}
        }
    }
}

/// Collision group used by every rover collider; its filter is set to
/// everything *except* this group so the rover never contacts itself.
/// `pub` so sensor modules can build queries that exclude the rover.
pub const ROVER_GROUP: Group = Group::GROUP_1;

/// Reads keyboard input and drives the rover.
/// W/S = throttle (spin all wheels forward/reverse).
/// A/D = steering: front wheels swing one way, rear wheels swing the opposite
/// way (4WS opposite-phase). Releasing the key lets the motor spring back to 0.
fn drive(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut wheel_joints: Query<&mut ImpulseJoint, (With<Wheel>, Without<SteeringKnuckle>)>,
    mut knuckle_joints: Query<(&SteeringKnuckle, &mut ImpulseJoint), Without<Wheel>>,
    power: Res<power_cubes::PowerState>,
    chassis_res: Res<ChassisEntity>,
    chassis_vel_q: Query<&Velocity>,
) {
    // --- Throttle ---
    // No power → wheels can't spin. Steering still works so you can
    // straighten up before coming to rest.
    let throttle = if !power.has_power() {
        0.0
    } else if keyboard.pressed(KeyCode::KeyW) {
        10.0
    } else if keyboard.pressed(KeyCode::KeyS) {
        -10.0
    } else {
        0.0
    };

    for mut joint in wheel_joints.iter_mut() {
        if let TypedJoint::RevoluteJoint(ref mut revolute) = joint.data {
            revolute.set_motor_velocity(throttle, 50.0);
        }
    }

    // --- Steering ---
    // A = +angle on the front knuckles, -angle on the rear (opposite phase).
    // D mirrors that. No input → target 0 and the motor springs back to center.
    //
    // The commanded angle tapers gently with speed: full MAX_STEER at rest,
    // STEER_SPEED_FLOOR × MAX_STEER once you're at STEER_SPEED_REF and
    // above. Keeps low-speed turning sharp while preventing twitchy
    // over-steer when you're already moving.
    let speed = chassis_res
        .0
        .and_then(|e| chassis_vel_q.get(e).ok())
        .map(|v| v.linvel.length())
        .unwrap_or(0.0);
    let t = (speed / STEER_SPEED_REF).clamp(0.0, 1.0);
    let scale = 1.0 - (1.0 - STEER_SPEED_FLOOR) * t;
    let steer = if keyboard.pressed(KeyCode::KeyA) {
        MAX_STEER * scale
    } else if keyboard.pressed(KeyCode::KeyD) {
        -MAX_STEER * scale
    } else {
        0.0
    };

    for (knuckle, mut joint) in knuckle_joints.iter_mut() {
        let target = if knuckle.is_front { steer } else { -steer };
        if let TypedJoint::RevoluteJoint(ref mut revolute) = joint.data {
            revolute.set_motor_position(target, STEERING_STIFFNESS, STEERING_DAMPING);
        }
    }
}

/// Speed (m/s) at which the steering taper reaches its floor.
const STEER_SPEED_REF: f32 = 10.0;
/// Fraction of MAX_STEER applied at and above STEER_SPEED_REF.
/// 1.0 = no taper, 0.0 = full lock-out. 0.75 is subtle but noticeable.
const STEER_SPEED_FLOOR: f32 = 0.75;

/// Separate system that attaches joints. Runs after attach_colliders.
/// Looks for wheel entities that have a Collider but no ImpulseJoint yet.
/// Builds the steering linkage for each wheel:
///   chassis  --(revolute around Y, steering motor)-->  knuckle
///   knuckle  --(revolute around Z, throttle motor)-->  wheel
/// The knuckle is a small dummy body whose orientation about chassis-Y is
/// the steering angle. The wheel then spins about the knuckle's Z, which
/// is the chassis Z rotated by that steering angle — so the wheel rolls
/// in whichever direction the knuckle is currently pointing.
fn attach_joints(
    mut commands: Commands,
    wheels: Query<(Entity, &Name), (With<Collider>, Without<ImpulseJoint>, With<Wheel>)>,
    chassis_res: Res<ChassisEntity>,
    xforms: Query<&GlobalTransform>,
) {
    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok(chassis_xform) = xforms.get(chassis_id) else { return };
    let chassis_pos = chassis_xform.translation();
    let chassis_rot = chassis_xform.rotation();

    for (wheel_entity, name) in wheels.iter() {
        let name_str = name.as_str();
        let Some(&(_, offset)) = WHEEL_OFFSETS.iter().find(|(n, _)| *n == name_str) else {
            continue;
        };
        let is_front = offset.x < 0.0;

        // Place the hub/knuckle at the wheel's current world position so
        // Rapier doesn't have to yank constraints into shape on frame 1.
        let hub_world_pos = chassis_pos + chassis_rot * offset;

        // 1) chassis ↔ hub: PRISMATIC along chassis-local Y. This is the
        // suspension travel — the hub can slide up and down relative to
        // the chassis. A spring+damper motor holds the rover up against
        // gravity and absorbs bumps. With four independent springs each
        // wheel can find its own ground height, so the chassis pitches
        // and rolls over uneven terrain instead of leaving a wheel in
        // the air. It also removes the over-constraint jitter from
        // having four rigid contact points on a flat floor — only the
        // spring length is rigid, the four heights are free to compromise.
        let suspension = PrismaticJointBuilder::new(Vec3::Y)
            .local_anchor1(offset)
            .local_anchor2(Vec3::ZERO)
            .motor_position(0.0, SUSPENSION_STIFFNESS, SUSPENSION_DAMPING)
            .limits([-SUSPENSION_TRAVEL, SUSPENSION_TRAVEL]);

        let hub = commands
            .spawn((
                Transform::from_translation(hub_world_pos),
                RigidBody::Dynamic,
                AdditionalMassProperties::MassProperties(MassProperties {
                    local_center_of_mass: Vec3::ZERO,
                    mass: 3.0,
                    principal_inertia: Vec3::splat(0.5),
                    principal_inertia_local_frame: Quat::IDENTITY,
                }),
                Sleeping::disabled(),
                ImpulseJoint::new(chassis_id, suspension),
            ))
            .id();

        // 2) hub ↔ knuckle: REVOLUTE around hub-local Y (steering). The
        // prismatic joint above locks all rotation between chassis and
        // hub, so hub-Y == chassis-Y in the world.
        let steering_joint = RevoluteJointBuilder::new(Vec3::Y)
            .local_anchor1(Vec3::ZERO)
            .local_anchor2(Vec3::ZERO)
            .motor_position(0.0, STEERING_STIFFNESS, STEERING_DAMPING)
            .limits([-MAX_STEER, MAX_STEER]);

        let knuckle = commands
            .spawn((
                Transform::from_translation(hub_world_pos),
                RigidBody::Dynamic,
                AdditionalMassProperties::MassProperties(MassProperties {
                    local_center_of_mass: Vec3::ZERO,
                    mass: 3.0,
                    principal_inertia: Vec3::splat(0.5),
                    principal_inertia_local_frame: Quat::IDENTITY,
                }),
                Sleeping::disabled(),
                ImpulseJoint::new(hub, steering_joint),
                SteeringKnuckle { is_front },
            ))
            .id();

        // 3) knuckle ↔ wheel: REVOLUTE around knuckle-local Z (spin).
        let spin_joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(Vec3::ZERO)
            .local_anchor2(Vec3::ZERO)
            .motor_velocity(0.0, 50.0);

        commands.entity(wheel_entity).insert(
            ImpulseJoint::new(knuckle, spin_joint),
        );
    }
}

/// Marker on the wheel entities so the drive system can target the spin motor.
#[derive(Component)]
struct Wheel;

/// Marker + per-wheel orientation flag on the knuckle entities. `is_front`
/// determines the sign of the steering target (front and rear are
/// opposite-phase for tighter turns).
#[derive(Component)]
struct SteeringKnuckle {
    is_front: bool,
}

/// Steering geometry constants.
const MAX_STEER: f32 = 0.5;            // ~28.6° wheel travel each side
// PD-controller gains for the steering motor. Sized as a compromise:
// stiff enough that contact forces don't visibly toe the wheels in
// (which they did at K = 300), but soft enough that the chassis isn't
// yanked into the steering instead of the knuckle (which happened at
// K = 3000). With knuckle inertia ≈ 0.5 kg·m², critical damping is
// 2·√(K·I) ≈ 2·√600 ≈ 49; 70 sits slightly over-damped so the wheels
// don't oscillate when you release.
const STEERING_STIFFNESS: f32 = 2000.0;
const STEERING_DAMPING: f32 = 170.0;

/// Suspension geometry + PD gains.
///   - SUSPENSION_TRAVEL: how far the hub can slide each way (m).
///   - SUSPENSION_STIFFNESS: spring rate per wheel (N/m). Sized so the
///     rover sags ≈ 20 cm under its own weight at rest:
///     compression ≈ chassis_weight / (4 · K) ≈ (50 · 9.81)/(4·600) ≈ 0.20 m.
///   - SUSPENSION_DAMPING: shock-absorber rate (N·s/m). Critical damping
///     for the per-wheel sprung mass m ≈ 12.5 kg is 2·√(K·m) ≈ 173.
///     We sit at ratio ≈ 0.46 (well under-critical) so bumps bounce a
///     couple of cycles before settling — visibly springy. Natural
///     period ≈ 2π/√(K/m·(1-ζ²)) ≈ 1.0 s.
const SUSPENSION_TRAVEL: f32 = 0.35;
const SUSPENSION_STIFFNESS: f32 = 600.0;
const SUSPENSION_DAMPING: f32 = 80.0;

/// R = respawn upright at the rover's current XZ position (useful for
/// flipping back over without losing progress across the terrain).
/// Shift+R = respawn at the world origin (full reset).
fn respawn_rover(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    rover_query: Query<Entity, With<RoverRoot>>,
    chassis_xform_q: Query<&GlobalTransform>,
    asset_server: Res<AssetServer>,
    mut chassis_res: ResMut<ChassisEntity>,
    terrain: Option<Res<terrain_controls::TerrainState>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyR) { return; }

    let to_origin = keyboard.pressed(KeyCode::ShiftLeft)
        || keyboard.pressed(KeyCode::ShiftRight);

    // Origin respawn: place the rover above the landing pad at the
    // bottom of the giant crater (not world y = 1.5, which is way above
    // the floor now). Current-location respawn: keep XZ, just lift Y by
    // a clearance so a flipped rover lands upright.
    let pad_y = terrain
        .as_ref()
        .map(|t| t.height_at(0.0, 0.0))
        .unwrap_or(0.0);
    let spawn_xform = if to_origin {
        Transform::from_xyz(0.0, pad_y + ROVER_SPAWN_CLEARANCE, 0.0)
    } else if let Some(chassis_id) = chassis_res.0 {
        if let Ok(gxf) = chassis_xform_q.get(chassis_id) {
            let p = gxf.translation();
            Transform::from_xyz(p.x, p.y + ROVER_SPAWN_CLEARANCE, p.z)
        } else {
            Transform::from_xyz(0.0, pad_y + ROVER_SPAWN_CLEARANCE, 0.0)
        }
    } else {
        Transform::from_xyz(0.0, pad_y + ROVER_SPAWN_CLEARANCE, 0.0)
    };

    // Despawn the old rover and all its children
    for entity in rover_query.iter() {
        commands.entity(entity).despawn();
    }

    // Clear the chassis reference so attach_colliders/attach_joints re-run
    chassis_res.0 = None;

    commands.spawn((
        RoverRoot,
        SceneRoot(asset_server.load("rover_1.glb#Scene0")),
        spawn_xform,
    ));
}

/// Marker for the FPS counter text node so the update system can find it.
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

fn update_fps_text(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    let Ok(mut text) = query.single_mut() else { return };
    let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) else { return };
    let Some(fps) = fps_diag.smoothed() else { return };
    **text = format!("FPS: {:>5.1}", fps);
}

/// F1 toggles Rapier's wireframe debug renderer. Off by default because
/// drawing every heightfield triangle edge as a gizmo line tanks the framerate.
fn toggle_debug_render(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut ctx: ResMut<DebugRenderContext>,
) {
    if keyboard.just_pressed(KeyCode::F1) {
        ctx.enabled = !ctx.enabled;
    }
}

/// Which camera the player is currently using.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    Orbit,
    #[default]
    FollowBehind,
}

/// State for the follow-behind rig. yaw/pitch/distance describe the
/// camera's position in spherical coordinates in the chassis's local frame,
/// so the rig keeps its pose relative to the rover as the rover turns.
///
/// yaw = 0, pitch = 0 places the camera directly behind the chassis at
/// `distance` along chassis-local +X (front wheels are at -X, so back is +X).
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

/// C toggles between the free orbit camera and the follow-behind camera.
/// PanOrbitCamera is disabled while following so it doesn't consume mouse
/// input. On return to orbit mode we re-focus it on the chassis so it
/// doesn't snap back to the origin.
fn toggle_camera_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<CameraMode>,
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<&GlobalTransform, (Without<Camera3d>, With<RigidBody>)>,
    mut camera_q: Query<&mut PanOrbitCamera>,
) {
    if !keyboard.just_pressed(KeyCode::KeyC) { return; }
    *mode = match *mode {
        CameraMode::Orbit => CameraMode::FollowBehind,
        CameraMode::FollowBehind => CameraMode::Orbit,
    };
    let Ok(mut pan_orbit) = camera_q.single_mut() else { return };
    pan_orbit.enabled = matches!(*mode, CameraMode::Orbit);
    if matches!(*mode, CameraMode::Orbit) {
        if let Some(chassis_id) = chassis_res.0 {
            if let Ok(gxf) = chassis_q.get(chassis_id) {
                pan_orbit.target_focus = gxf.translation();
            }
        }
    }
}

/// Orbits the camera around the chassis in the chassis's local frame.
/// Left-click drag adjusts yaw/pitch, scroll adjusts distance. Runs in
/// PostUpdate so it overrides any Transform PanOrbitCamera wrote earlier.
fn follow_camera(
    time: Res<Time>,
    mode: Res<CameraMode>,
    chassis_res: Res<ChassisEntity>,
    // The chassis is a child of the rover-root entity in the glTF hierarchy,
    // so its local Transform is relative to that parent — not world space.
    // We need GlobalTransform here, otherwise after a respawn at a non-origin
    // location the camera lerps toward (0,0,0)-relative coordinates.
    chassis_q: Query<&GlobalTransform, (Without<Camera3d>, With<RigidBody>)>,
    mut camera_q: Query<(&mut Transform, &mut FollowCamera), With<Camera3d>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
    panel_q: Query<&bevy::ui::RelativeCursorPosition, With<TerrainPanel>>,
) {
    // Drain the readers in orbit mode so we don't accumulate a backlog of
    // mouse deltas to apply on next toggle.
    if !matches!(*mode, CameraMode::FollowBehind) {
        mouse_motion.read().count();
        scroll.read().count();
        return;
    }

    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok(chassis_gxf) = chassis_q.get(chassis_id) else { return };
    let chassis_pos = chassis_gxf.translation();
    let chassis_rot = chassis_gxf.rotation();
    let Ok((mut cam_xform, mut follow)) = camera_q.single_mut() else { return };

    // Ignore mouse input that lands on the terrain panel so dragging a
    // button or hovering the panel doesn't also rotate the view.
    let over_ui = cursor_over_terrain_panel(&panel_q);

    // Accumulate mouse drag (left button held) into yaw/pitch.
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
        follow.pitch = (follow.pitch + dy * follow.sensitivity)
            .clamp(-1.4, 1.4);
    }

    // Scroll wheel zooms (log-scale for smooth feel) — also gated by
    // cursor-over-panel so scrolling the UI doesn't zoom the rover view.
    let mut s = 0.0_f32;
    if !over_ui {
        for ev in scroll.read() {
            s += ev.y;
        }
    } else {
        scroll.read().count();
    }
    if s != 0.0 {
        follow.distance = (follow.distance * (1.0 - s * follow.zoom_sensitivity))
            .clamp(2.0, 100.0);
    }

    // Spherical position in chassis-local space. At yaw=pitch=0 the camera
    // sits at (+distance, 0, 0) — directly behind the rover.
    let cp = follow.pitch.cos();
    let local_offset = Vec3::new(
        follow.distance * cp * follow.yaw.cos(),
        follow.distance * follow.pitch.sin(),
        follow.distance * cp * follow.yaw.sin(),
    );

    // Project the chassis's forward vector onto the world XZ plane and use
    // only that yaw to place the camera. This way the camera tracks the
    // rover's heading but never inherits its roll or pitch — driving over a
    // crater rim or rolling onto its side won't tilt the camera image.
    let chassis_fwd = chassis_rot * Vec3::NEG_X;
    let flat = Vec3::new(chassis_fwd.x, 0.0, chassis_fwd.z);
    let yaw_rot = if flat.length_squared() > 1e-4 {
        let flat = flat.normalize();
        // Q(θ) around world Y satisfies Q * NEG_X = (-cos θ, 0, sin θ),
        // so θ = atan2(flat.z, -flat.x).
        Quat::from_rotation_y(flat.z.atan2(-flat.x))
    } else {
        Quat::IDENTITY
    };

    let world_offset = yaw_rot * local_offset;
    let target_pos = chassis_pos + world_offset;

    // Light translation smoothing so suspension bounces glide.
    let alpha = 1.0 - (-time.delta_secs() * 12.0).exp();
    cam_xform.translation = cam_xform.translation.lerp(target_pos, alpha);

    let look_target = chassis_pos + Vec3::Y * 0.5;
    cam_xform.look_at(look_target, Vec3::Y);
}

/// Listens for `RelaunchEvent` fired by the game-over splash's button.
/// Despawns the old rover and clears the chassis reference; the next
/// `spawn_initial_rover` tick will then re-spawn at the landing pad.
/// (Power has already been topped up by the button handler.)
fn handle_relaunch_event(
    mut commands: Commands,
    mut events: MessageReader<power_cubes::RelaunchEvent>,
    rover_q: Query<Entity, With<RoverRoot>>,
    mut chassis_res: ResMut<ChassisEntity>,
) {
    if events.read().count() == 0 {
        return;
    }
    for entity in rover_q.iter() {
        commands.entity(entity).despawn();
    }
    chassis_res.0 = None;
}
