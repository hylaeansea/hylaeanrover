//! Glowing blue power cubes that spawn randomly across the terrain.
//!
//! Spawn timing follows a homogeneous Poisson process — the interval
//! between successive spawns is drawn from `Exp(λ)` (i.e. `−ln(U)/λ`
//! for `U ~ Uniform(0,1)`). At `SPAWN_LAMBDA = 0.2 /s` the long-run
//! average is one spawn every 5 seconds.
//!
//! Each cube is a dynamic Rapier rigid body with a cuboid collider, so
//! it falls onto the terrain, tumbles, and gets shoved by the rover.
//! The visual material has an HDR-range emissive value so the bloom
//! post-process kicks in around it (see the camera setup in `main.rs`).

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use rand::Rng;

use crate::terrain_controls::TerrainState;

/// Marker on every spawned cube so other systems (pickup, despawn-on-far
/// later) can find them with `Query<..., With<PowerCube>>`.
#[derive(Component)]
pub struct PowerCube;

#[derive(Resource, Default)]
struct PowerCubeAssets {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct PoissonSpawner {
    time_to_next: f32,
}

impl Default for PoissonSpawner {
    fn default() -> Self {
        // First cube comes shortly after startup so you don't sit there
        // waiting through the first Poisson interval.
        Self { time_to_next: 1.0 }
    }
}

/// Expected spawns per second. 0.2 → ~1 cube every 5 seconds on average.
const SPAWN_LAMBDA: f32 = 0.2;
/// Half-extent of the cube collider/mesh (meters).
const CUBE_HALF_EXTENT: f32 = 0.35;
/// Drop the cube from this height; physics handles the rest.
const SPAWN_HEIGHT: f32 = 40.0;
/// Stay this far in from the terrain edge so cubes never spawn off the map.
const EDGE_MARGIN_FRAC: f32 = 0.9;

pub struct PowerCubesPlugin;

impl Plugin for PowerCubesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PowerCubeAssets>()
            .init_resource::<PoissonSpawner>()
            .add_systems(Startup, setup_assets)
            .add_systems(Update, spawn_power_cubes);
    }
}

fn setup_assets(
    mut assets: ResMut<PowerCubeAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    assets.mesh = meshes.add(Cuboid::new(
        CUBE_HALF_EXTENT * 2.0,
        CUBE_HALF_EXTENT * 2.0,
        CUBE_HALF_EXTENT * 2.0,
    ));
    assets.material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.05, 0.25, 0.95),
        // emissive is in linear RGB; values > 1.0 push pixels above the
        // bloom threshold so the cube radiates blue light.
        emissive: LinearRgba::new(0.2, 1.5, 8.0, 1.0),
        perceptual_roughness: 0.35,
        metallic: 0.1,
        ..default()
    });
}

fn spawn_power_cubes(
    mut commands: Commands,
    time: Res<Time>,
    mut spawner: ResMut<PoissonSpawner>,
    assets: Res<PowerCubeAssets>,
    terrain: Option<Res<TerrainState>>,
) {
    spawner.time_to_next -= time.delta_secs();
    if spawner.time_to_next > 0.0 {
        return;
    }

    let mut rng = rand::thread_rng();
    // Exponential interarrival → Poisson process. Guard against u = 0
    // which would blow up ln(0).
    let u: f32 = rng.gen_range(f32::EPSILON..1.0_f32);
    spawner.time_to_next = -u.ln() / SPAWN_LAMBDA;

    // Pick a random spot inside the terrain footprint. If the terrain
    // resource isn't ready yet (very early frames), skip this spawn.
    let Some(terrain) = terrain.as_ref() else { return };
    let extent = terrain.size * EDGE_MARGIN_FRAC;
    let x: f32 = rng.gen_range(-extent..extent);
    let z: f32 = rng.gen_range(-extent..extent);

    // Random initial spin so cubes look organic as they fall.
    let spin = Vec3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );

    commands.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.material.clone()),
        Transform::from_xyz(x, SPAWN_HEIGHT, z),
        RigidBody::Dynamic,
        Collider::cuboid(CUBE_HALF_EXTENT, CUBE_HALF_EXTENT, CUBE_HALF_EXTENT),
        Friction::coefficient(0.5),
        Restitution::coefficient(0.2),
        Velocity {
            linvel: Vec3::ZERO,
            angvel: spin,
        },
        PowerCube,
    ));
}
