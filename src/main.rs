use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;

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
        // Draws wireframes around colliders so we can see if they match our meshes
        // Remove this once we're happy with the collider shapes
        .add_plugins(RapierDebugRenderPlugin::default())
        // Initialize the chassis resource as empty — attach_physics will fill it
        .init_resource::<ChassisEntity>()
        // Run `setup` once at startup, before the first frame
        .add_systems(Startup, setup)
        // Two-phase physics setup:
        // 1. attach_colliders finds glTF entities by name and adds colliders
        // 2. attach_joints runs after, connecting wheels to chassis with hinges
        .add_systems(Update, (attach_colliders, attach_joints).chain())
        .add_systems(Update, drive)
        .add_systems(Update, respawn_rover)
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

    // --- Ground Plane ---
    // A 50x50 flat surface at the origin. We add RigidBody::Fixed so Rapier knows
    // this doesn't move, and a halfspace collider (infinite plane pointing up).
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.3))),
        RigidBody::Fixed,
        Collider::halfspace(Vect::Y).unwrap(),
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
        Transform::from_xyz(0.0, 5.0, 0.0),
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
        Transform::from_xyz(0.0, 5.0, 0.0),
    ));
}
