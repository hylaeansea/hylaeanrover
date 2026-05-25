//! Survey beacons the player drops from the back of the rover.
//!
//! Press **B** while driving and the next frame a beacon glTF is
//! instantiated `BEACON_BEHIND_DISTANCE` meters behind the chassis
//! (in chassis-local +X, since the chassis's forward is −X) and
//! settled onto the local terrain height. The beacon is static —
//! no rigid body, no collider — so it just stays where it was placed.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::minerals::MineralMaps;
use crate::power_cubes::RelaunchEvent;
use crate::reward::RewardState;
use crate::terrain_controls::TerrainState;
use crate::ChassisEntity;

/// Where behind the chassis (chassis-local +X) the beacon spawns, in
/// meters. Far enough to clear the back wheels.
const BEACON_BEHIND_DISTANCE: f32 = 4.0;

/// Collider dimensions (cylinder along world-Y, sitting on top of the
/// beacon's transform origin so it matches the glTF model planted on
/// the ground). The collider is deliberately taller than the visible
/// post — extending past the rover's underbody height — so the chassis
/// can never sweep over the top of it between two physics frames.
const BEACON_HALF_HEIGHT: f32 = 1.0;
const BEACON_RADIUS: f32 = 0.3;

/// Marker on every spawned beacon so future systems (count, recall,
/// re-survey) can find them.
#[derive(Component)]
pub struct Beacon;

/// Caches the loaded glTF scene so each press is a cheap clone of a
/// handle rather than another asset load.
#[derive(Resource, Default)]
struct BeaconAssets {
    scene: Option<Handle<Scene>>,
}

pub struct BeaconsPlugin;

impl Plugin for BeaconsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BeaconAssets>()
            .add_systems(Startup, load_beacon_asset)
            .add_systems(Update, (place_beacon_on_input, despawn_beacons_on_relaunch));
    }
}

fn despawn_beacons_on_relaunch(
    mut commands: Commands,
    mut events: MessageReader<RelaunchEvent>,
    beacons_q: Query<Entity, With<Beacon>>,
) {
    if events.read().count() == 0 {
        return;
    }
    for entity in beacons_q.iter() {
        commands.entity(entity).despawn();
    }
}

fn load_beacon_asset(asset_server: Res<AssetServer>, mut assets: ResMut<BeaconAssets>) {
    assets.scene = Some(asset_server.load("beacon_1.glb#Scene0"));
}

fn place_beacon_on_input(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    chassis_res: Res<ChassisEntity>,
    chassis_q: Query<&GlobalTransform>,
    terrain: Option<Res<TerrainState>>,
    assets: Res<BeaconAssets>,
    maps: Res<MineralMaps>,
    mut reward: ResMut<RewardState>,
) {
    if !keyboard.just_pressed(KeyCode::KeyB) {
        return;
    }
    let Some(chassis_id) = chassis_res.0 else { return };
    let Ok(chassis_gxf) = chassis_q.get(chassis_id) else { return };
    let Some(scene) = assets.scene.as_ref() else { return };

    // Project the chassis forward vector onto the world XZ plane to get
    // a pure-yaw rotation; we want the beacon upright (Y world up) but
    // facing the same heading as the rover.
    let chassis_pos = chassis_gxf.translation();
    let chassis_rot = chassis_gxf.rotation();
    let chassis_fwd = chassis_rot * Vec3::NEG_X;
    let flat = Vec3::new(chassis_fwd.x, 0.0, chassis_fwd.z);
    let yaw_rot = if flat.length_squared() > 1e-4 {
        let flat = flat.normalize();
        Quat::from_rotation_y(flat.z.atan2(-flat.x))
    } else {
        Quat::IDENTITY
    };

    // "Behind" the chassis is its local +X direction (since the front
    // wheels are at chassis-local x = -2.49).
    let world_offset = yaw_rot * Vec3::new(BEACON_BEHIND_DISTANCE, 0.0, 0.0);
    let mut beacon_pos = chassis_pos + world_offset;

    // Snap to the terrain so the beacon sits on the ground regardless
    // of the rover's suspension state.
    if let Some(terrain) = terrain {
        beacon_pos.y = terrain.height_at(beacon_pos.x, beacon_pos.z);
    }

    // Try to spend a beacon from the budget. If the player has none
    // left, refuse to place — same effect as the key being dead.
    if reward
        .try_credit_beacon(&maps, beacon_pos.x, beacon_pos.z)
        .is_none()
    {
        return;
    }

    commands.spawn((
        Beacon,
        SceneRoot(scene.clone()),
        Transform {
            translation: beacon_pos,
            rotation: yaw_rot,
            scale: Vec3::ONE,
        },
        // Static body so the beacon stays planted in the ground even if
        // the rover slams into it. Compound + Y offset places the
        // cylinder *above* the entity origin so it lines up with the
        // visible model rather than half-burying the collider.
        RigidBody::Fixed,
        Collider::compound(vec![(
            Vect::new(0.0, BEACON_HALF_HEIGHT, 0.0),
            Quat::IDENTITY,
            Collider::cylinder(BEACON_HALF_HEIGHT, BEACON_RADIUS),
        )]),
    ));
}
