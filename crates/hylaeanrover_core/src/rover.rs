//! Rover physics: chassis + 4-wheel suspension + drive train + respawn.
//!
//! The visual rover is normally loaded from a glTF (`assets/rover_1.glb`),
//! but for headless / RL use we can spawn the same named entities as
//! primitives so no rendering / asset pipeline is needed. `RoverPlugin`
//! takes a `RoverSpawnMode` selecting which path to use.
//!
//! `attach_colliders` + `attach_joints` are the workhorses — both ignore
//! the spawn mode entirely, they just look for entities named "chassis"
//! and "wheel_*" and wire physics + suspension + steering onto them.
//! That makes the two spawn modes effectively interchangeable.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::power_cubes::{self, PowerState, RelaunchEvent};
use crate::terrain_controls;

// ---- Public types --------------------------------------------------------

/// Marker on the rover root entity so we can find it to despawn.
#[derive(Component)]
pub struct RoverRoot;

/// Stores the chassis Entity so wheels can create joints back to it,
/// the follow camera can target it, and the power system can read its
/// position.
#[derive(Resource, Default)]
pub struct ChassisEntity(pub Option<Entity>);

/// Collision group used by every rover collider; its filter is set to
/// everything *except* this group so the rover never contacts itself.
pub const ROVER_GROUP: Group = Group::GROUP_1;

/// Marker on the wheel entities so the drive system can target the spin motor.
#[derive(Component)]
pub struct Wheel;

/// Marker + per-wheel orientation flag on the knuckle entities. `is_front`
/// determines the sign of the steering target (front and rear are
/// opposite-phase for tighter turns).
#[derive(Component)]
pub struct SteeringKnuckle {
    pub is_front: bool,
}

/// How the rover's visible entities are created.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum RoverSpawnMode {
    /// Load `assets/rover_1.glb`. Requires `AssetServer` +
    /// `ScenePlugin` to be present in the app.
    #[default]
    Gltf,
    /// Spawn primitive named entities directly — no asset / render
    /// dependency. Use this for headless RL.
    Primitives,
}

/// Stores the chosen `RoverSpawnMode` as a resource so other plugins
/// (e.g. `beacons` deciding whether to load the beacon glTF) can branch
/// on it without touching the rover module's internals.
#[derive(Resource, Clone, Copy)]
pub struct SpawnConfig(pub RoverSpawnMode);

impl SpawnConfig {
    pub fn is_headless(&self) -> bool {
        matches!(self.0, RoverSpawnMode::Primitives)
    }
}

// ---- Constants -----------------------------------------------------------

/// Clearance above the local terrain at which the rover is spawned.
pub const ROVER_SPAWN_CLEARANCE: f32 = 1.5;

/// Wheel positions relative to the chassis origin (Bevy coordinates).
const WHEEL_OFFSETS: &[(&str, Vec3)] = &[
    ("wheel_fl", Vec3::new(-2.49, -0.66, 1.78)),
    ("wheel_fr", Vec3::new(-2.49, -0.66, -1.73)),
    ("wheel_bl", Vec3::new(2.10, -0.66, 1.78)),
    ("wheel_br", Vec3::new(2.10, -0.66, -1.73)),
];

/// Steering geometry.
pub const MAX_STEER: f32 = 0.5; // ~28.6° each side
pub const STEERING_STIFFNESS: f32 = 2000.0;
pub const STEERING_DAMPING: f32 = 170.0;

/// Steering-taper params: full lock at rest, STEER_SPEED_FLOOR × MAX
/// at and above STEER_SPEED_REF m/s.
const STEER_SPEED_REF: f32 = 10.0;
const STEER_SPEED_FLOOR: f32 = 0.75;

/// Suspension geometry + gains.
pub const SUSPENSION_TRAVEL: f32 = 0.35;
pub const SUSPENSION_STIFFNESS: f32 = 600.0;
pub const SUSPENSION_DAMPING: f32 = 80.0;

// ---- Plugin --------------------------------------------------------------

/// Bundles the rover's physics + drive + respawn logic. The
/// `spawn_mode` controls only the *visible* entity hierarchy — the
/// collider/joint setup is identical either way.
pub struct RoverPlugin {
    pub spawn_mode: RoverSpawnMode,
}

impl Default for RoverPlugin {
    fn default() -> Self {
        Self { spawn_mode: RoverSpawnMode::Gltf }
    }
}

impl RoverPlugin {
    pub fn headless() -> Self {
        Self { spawn_mode: RoverSpawnMode::Primitives }
    }
}

impl Plugin for RoverPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChassisEntity>()
            .insert_resource(SpawnConfig(self.spawn_mode))
            .add_systems(
                Update,
                (
                    spawn_initial_rover,
                    (attach_colliders, attach_joints).chain(),
                    drive,
                    respawn_rover,
                    handle_relaunch_event,
                ),
            );
    }
}

// ---- Spawn / respawn -----------------------------------------------------

fn spawn_initial_rover(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    terrain: Option<Res<terrain_controls::TerrainState>>,
    existing: Query<(), With<RoverRoot>>,
    cfg: Res<SpawnConfig>,
) {
    if !existing.is_empty() {
        return;
    }
    let Some(terrain) = terrain else { return };
    let pad_height = terrain.height_at(0.0, 0.0);
    let xform = Transform::from_xyz(0.0, pad_height + ROVER_SPAWN_CLEARANCE, 0.0);

    match cfg.0 {
        RoverSpawnMode::Gltf => {
            // The glTF path produces a child hierarchy with named
            // entities ("chassis", "wheel_fl", …) that
            // `attach_colliders` then picks up.
            if let Some(asset_server) = asset_server {
                commands.spawn((
                    RoverRoot,
                    SceneRoot(asset_server.load("rover_1.glb#Scene0")),
                    xform,
                ));
            }
        }
        RoverSpawnMode::Primitives => spawn_rover_primitives(&mut commands, xform),
    }
}

/// Spawns the rover as bare named entities — same names the glTF would
/// produce, so `attach_colliders` does the rest of the work uniformly.
fn spawn_rover_primitives(commands: &mut Commands, root_xform: Transform) {
    let root = commands
        .spawn((RoverRoot, root_xform, Visibility::default()))
        .id();

    // Chassis sits at the root entity's local origin in the glTF
    // hierarchy, so we place its local Transform at the origin and let
    // the root's transform propagate.
    commands.spawn((
        Name::new("chassis"),
        Transform::IDENTITY,
        ChildOf(root),
    ));
    for (name, _offset) in WHEEL_OFFSETS {
        // Wheels sit at the chassis's local origin pre-physics — the
        // joint setup in `attach_joints` translates them to their
        // suspension anchor positions before the first physics step.
        commands.spawn((
            Name::new(*name),
            Transform::IDENTITY,
            ChildOf(root),
        ));
    }
}

/// R = respawn upright at the rover's current XZ position.
/// Shift+R = respawn at the world origin.
fn respawn_rover(
    mut commands: Commands,
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    rover_query: Query<Entity, With<RoverRoot>>,
    chassis_xform_q: Query<&GlobalTransform>,
    asset_server: Option<Res<AssetServer>>,
    mut chassis_res: ResMut<ChassisEntity>,
    terrain: Option<Res<terrain_controls::TerrainState>>,
    cfg: Res<SpawnConfig>,
) {
    let Some(keyboard) = keyboard else { return };
    if !keyboard.just_pressed(KeyCode::KeyR) { return; }

    let to_origin = keyboard.pressed(KeyCode::ShiftLeft)
        || keyboard.pressed(KeyCode::ShiftRight);

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

    for entity in rover_query.iter() {
        commands.entity(entity).despawn();
    }
    chassis_res.0 = None;

    match cfg.0 {
        RoverSpawnMode::Gltf => {
            if let Some(asset_server) = asset_server {
                commands.spawn((
                    RoverRoot,
                    SceneRoot(asset_server.load("rover_1.glb#Scene0")),
                    spawn_xform,
                ));
            }
        }
        RoverSpawnMode::Primitives => spawn_rover_primitives(&mut commands, spawn_xform),
    }
}

/// Listens for `RelaunchEvent` from the game-over modal: despawn the
/// rover and clear the chassis reference so `spawn_initial_rover`
/// picks up where it left off next tick.
fn handle_relaunch_event(
    mut commands: Commands,
    mut events: MessageReader<RelaunchEvent>,
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

// ---- Physics: colliders + joints -----------------------------------------

fn attach_colliders(
    mut commands: Commands,
    query: Query<(Entity, &Name), Without<Collider>>,
    mut chassis_res: ResMut<ChassisEntity>,
) {
    // See main.rs:195 for the rationale — the rover's parts must never
    // contact themselves (joints already constrain them) so they live
    // in a self-excluding collision group.
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
                    Sleeping::disabled(),
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
        let hub_world_pos = chassis_pos + chassis_rot * offset;

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

        let spin_joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(Vec3::ZERO)
            .local_anchor2(Vec3::ZERO)
            .motor_velocity(0.0, 50.0);

        commands.entity(wheel_entity).insert(
            ImpulseJoint::new(knuckle, spin_joint),
        );
    }
}

// ---- Drive ---------------------------------------------------------------

/// Reads `RoverAction` (when present) for RL-driven control; otherwise
/// falls back to W/A/S/D keyboard input. The `RoverAction` resource is
/// the cleanest seam for the PyO3 env to inject actions each step.
///
/// Set `drop_beacon = true` on the frame you want to fire a beacon (the
/// place-beacon system consumes the flag and resets it to false).
#[derive(Resource, Default, Clone, Copy)]
pub struct RoverAction {
    /// -1..=+1, +1 = forward
    pub throttle: f32,
    /// -1..=+1, +1 = steer right (compass convention)
    pub steering: f32,
    /// One-shot edge-triggered beacon drop request.
    pub drop_beacon: bool,
}

fn drive(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    action: Option<Res<RoverAction>>,
    mut wheel_joints: Query<&mut ImpulseJoint, (With<Wheel>, Without<SteeringKnuckle>)>,
    mut knuckle_joints: Query<(&SteeringKnuckle, &mut ImpulseJoint), Without<Wheel>>,
    power: Res<PowerState>,
    chassis_res: Res<ChassisEntity>,
    chassis_vel_q: Query<&Velocity>,
) {
    // Action-driven (RL) takes precedence over keyboard when the
    // resource is present. The keyboard path is what the game binary
    // uses; the env populates RoverAction each step. In headless mode
    // (no InputPlugin) keyboard is None and we drive from action only.
    let (throttle_in, steer_in) = match (&action, &keyboard) {
        (Some(a), _) if a.throttle != 0.0 || a.steering != 0.0 => (a.throttle, a.steering),
        (_, Some(kb)) => {
            let t = if kb.pressed(KeyCode::KeyW) {
                1.0
            } else if kb.pressed(KeyCode::KeyS) {
                -1.0
            } else {
                0.0
            };
            let s = if kb.pressed(KeyCode::KeyA) {
                1.0
            } else if kb.pressed(KeyCode::KeyD) {
                -1.0
            } else {
                0.0
            };
            (t, s)
        }
        _ => (0.0, 0.0),
    };

    let throttle = if !power.has_power() { 0.0 } else { throttle_in * 10.0 };

    for mut joint in wheel_joints.iter_mut() {
        if let TypedJoint::RevoluteJoint(ref mut revolute) = joint.data {
            revolute.set_motor_velocity(throttle, 50.0);
        }
    }

    let speed = chassis_res
        .0
        .and_then(|e| chassis_vel_q.get(e).ok())
        .map(|v| v.linvel.length())
        .unwrap_or(0.0);
    let t = (speed / STEER_SPEED_REF).clamp(0.0, 1.0);
    let scale = 1.0 - (1.0 - STEER_SPEED_FLOOR) * t;
    let steer = steer_in * MAX_STEER * scale;

    for (knuckle, mut joint) in knuckle_joints.iter_mut() {
        let target = if knuckle.is_front { steer } else { -steer };
        if let TypedJoint::RevoluteJoint(ref mut revolute) = joint.data {
            revolute.set_motor_position(target, STEERING_STIFFNESS, STEERING_DAMPING);
        }
    }
}

// re-export so callers can `use power_cubes::*` from outside
#[allow(unused_imports)]
pub use power_cubes::PowerCube;
