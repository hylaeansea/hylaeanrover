use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;

mod terrain;
use terrain::{LunarTerrain, LunarTerrainConfig};

fn main() {
    App::new()
        // DefaultPlugins gives us a window, renderer, input, asset loading —
        // everything needed to actually see something on screen
        .add_plugins(DefaultPlugins)
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
        // Initialize the chassis resource as empty — attach_physics will fill it
        .init_resource::<ChassisEntity>()
        // Run `setup` once at startup, before the first frame
        .add_systems(Startup, (setup, setup_fps_text))
        // Two-phase physics setup:
        // 1. attach_colliders finds glTF entities by name and adds colliders
        // 2. attach_joints runs after, connecting wheels to chassis with hinges
        .add_systems(Update, (attach_colliders, attach_joints).chain())
        .add_systems(Update, drive)
        .add_systems(Update, respawn_rover)
        .add_systems(Update, (update_fps_text, toggle_debug_render))
        .run();
}

fn setup(
    // Commands lets us spawn entities into the world
    mut commands: Commands,
    // AssetServer loads files from the assets/ folder
    asset_server: Res<AssetServer>,
    // Assets<Mesh> and Assets<StandardMaterial> let us create geometry at runtime
    // ResMut because we're adding new assets, not just reading
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

    // --- Lunar Terrain ---
    // Procedurally generated heightmap: rolling regolith with impact craters.
    // The visual mesh and the Rapier heightfield collider share the same height
    // grid, so what you drive on is exactly what you see.
    let terrain = LunarTerrain::generate(LunarTerrainConfig::default());
    let n = terrain.resolution;
    let span = 2.0 * terrain.size;
    // Rapier's heightfield convention: rows are the Z axis, columns are the X
    // axis, and the Vec is read in column-major order. Our heights are stored
    // row-major as `heights[z * n + x]`, so we reorder into [x_outer, z_inner].
    let mut hf_heights = Vec::with_capacity(n * n);
    for x in 0..n {
        for z in 0..n {
            hf_heights.push(terrain.heights[z * n + x]);
        }
    }
    let heightfield = Collider::heightfield(
        hf_heights,
        n, // num_rows -> Z
        n, // num_cols -> X
        Vect::new(span, 1.0, span),
    );
    commands.spawn((
        Mesh3d(meshes.add(terrain.build_mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.3),
            perceptual_roughness: 0.95,
            metallic: 0.0,
            reflectance: 0.2,
            ..default()
        })),
        RigidBody::Fixed,
        heightfield,
    ));

    // --- Rover ---
    // asset_server.load() looks in the assets/ folder. The #Scene0 fragment
    // selects the first scene from the glTF file — this is your entire Blender
    // scene graph (chassis, wheels, etc.) preserved as a hierarchy.
    // SceneRoot spawns all those nodes as children of this entity.
    // Lift the rover up so it spawns above the ground and falls onto it
    commands.spawn((
        RoverRoot,
        SceneRoot(asset_server.load("rover_1.glb#Scene0")),
        Transform::from_xyz(0.0, 1.5, 0.0),
    ));
}

/// Marker component on the rover root entity so we can find it to despawn
#[derive(Component)]
struct RoverRoot;

/// Stores the chassis Entity so wheels can create joints back to it.
/// Option<Entity> because it starts as None until the chassis is found.
#[derive(Resource, Default)]
struct ChassisEntity(Option<Entity>);

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
                    // ExternalImpulse lets us apply steering kicks from the drive system
                    ExternalImpulse::default(),
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
                    // High friction for knobby off-road tires — more grip, less sliding
                    Friction::coefficient(1.0),
                ));
            }
            _ => {}
        }
    }
}

/// Reads keyboard input and drives the rover.
/// W/S = throttle (forward/reverse), A/D = steering (torque on chassis)
fn drive(
    keyboard: Res<ButtonInput<KeyCode>>,
    // Query all wheel joints — we'll update their motor velocity
    mut wheels: Query<&mut ImpulseJoint>,
    // Query the chassis for applying steering torque
    chassis_res: Res<ChassisEntity>,
    mut ext_impulses: Query<&mut ExternalImpulse>,
) {
    // --- Throttle ---
    // W = forward (negative velocity because of wheel spin direction),
    // S = reverse. 0 if neither pressed — wheels coast to a stop via motor damping.
    let throttle = if keyboard.pressed(KeyCode::KeyW) { 10.0 }
        else if keyboard.pressed(KeyCode::KeyS) { -10.0 }
        else { 0.0 };

    // Update every wheel joint's motor target velocity
    for mut joint in wheels.iter_mut() {
        // Pattern match to get the inner RevoluteJoint from the TypedJoint enum,
        // then set the motor velocity. This tells Rapier "spin this joint toward
        // this velocity, applying up to 50 units of torque to get there"
        if let TypedJoint::RevoluteJoint(ref mut revolute) = joint.data {
            revolute.set_motor_velocity(throttle, 50.0);
        }
    }

    // --- Steering ---
    // A/D applies a torque around Y (up axis) to the chassis body.
    // This rotates the whole rover left/right. Hacky but effective for V1.
    // ExternalImpulse is an instant angular kick, applied once per frame.
    // Much more responsive than ExternalForce for steering.
    // Flip steering when reversing — A should always turn the rover "left"
    // relative to its direction of travel
    let direction = if throttle >= 0.0 { 1.0 } else { -1.0 };
    let steer_impulse = if keyboard.pressed(KeyCode::KeyA) { 5.0 * direction }
        else if keyboard.pressed(KeyCode::KeyD) { -5.0 * direction }
        else { 0.0 };

    if let Some(chassis_id) = chassis_res.0 {
        if let Ok(mut impulse) = ext_impulses.get_mut(chassis_id) {
            impulse.torque_impulse = Vec3::new(0.0, steer_impulse, 0.0);
        }
    }
}

/// Separate system that attaches joints. Runs after attach_colliders.
/// Looks for wheel entities that have a Collider but no ImpulseJoint yet.
/// This handles the race condition where wheels might get colliders before
/// the chassis is found — joints are added on the next frame once both exist.
fn attach_joints(
    mut commands: Commands,
    wheels: Query<(Entity, &Name), (With<Collider>, Without<ImpulseJoint>)>,
    chassis_res: Res<ChassisEntity>,
) {
    // Can't create joints until we know the chassis entity
    let Some(chassis_id) = chassis_res.0 else { return };

    for (entity, name) in wheels.iter() {
        let name_str = name.as_str();

        if let Some(&(_, offset)) = WHEEL_OFFSETS.iter().find(|(n, _)| *n == name_str) {
            // RevoluteJoint = a hinge. The wheel can spin around one axis
            // but is otherwise locked to the chassis.
            // Vec3::Z = the spin axis (axle direction, side-to-side)
            // local_anchor1 = where on the chassis the joint connects
            // local_anchor2 = where on the wheel (its center)
            let joint = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(offset)
                .local_anchor2(Vec3::ZERO)
                // Motor: target velocity 0 (updated by drive system), max force 50
                // The factor controls how hard the motor pushes to reach target speed
                .motor_velocity(0.0, 50.0);

            commands.entity(entity).insert(
                ImpulseJoint::new(chassis_id, joint),
            );
        }
    }
}

/// Press R to despawn the rover and spawn a fresh one at the starting position.
/// DespawnRecursive removes the entity and all its children (wheels, meshes, etc.)
fn respawn_rover(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    rover_query: Query<Entity, With<RoverRoot>>,
    asset_server: Res<AssetServer>,
    mut chassis_res: ResMut<ChassisEntity>,
) {
    if !keyboard.just_pressed(KeyCode::KeyR) { return; }

    // Despawn the old rover and all its children
    for entity in rover_query.iter() {
        commands.entity(entity).despawn();
    }

    // Clear the chassis reference so attach_colliders/attach_joints re-run
    chassis_res.0 = None;

    // Spawn a fresh rover
    commands.spawn((
        RoverRoot,
        SceneRoot(asset_server.load("rover_1.glb#Scene0")),
        Transform::from_xyz(0.0, 1.5, 0.0),
    ));
}

/// Marker for the FPS counter text node so the update system can find it.
#[derive(Component)]
struct FpsText;

fn setup_fps_text(mut commands: Commands) {
    commands.spawn((
        Text::new("FPS: --"),
        TextFont { font_size: 18.0, ..default() },
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
