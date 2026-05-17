use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Procedurally generated lunar terrain heightmap.
///
/// Spans `[-size, size]` in X and Z. `heights` is stored row-major as
/// `heights[z * resolution + x]`. This layout is also column-major in
/// Rapier's heightfield convention (rows = X, cols = Z), so the same Vec
/// can be handed to `Collider::heightfield` without reordering.
#[derive(Clone)]
pub struct LunarTerrain {
    pub size: f32,
    pub resolution: usize,
    pub heights: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct LunarTerrainConfig {
    pub seed: u64,
    pub size: f32,
    pub resolution: usize,
    pub crater_count: usize,
}

impl Default for LunarTerrainConfig {
    fn default() -> Self {
        // 9× the previous half-extent → ~5 km arena. The terrain is set
        // up like a single giant impact crater: a roughly flat floor for
        // the inner ~85% of the radius, with a tall steep rim around the
        // edge that the rover physically can't climb out of (see
        // `stamp_bowl_rim`). Resolution chosen to keep cell size ≈ 6 m so
        // the heightfield collider stays tractable on this much larger
        // area without smearing out the local crater detail.
        Self {
            seed: 42,
            size: 2475.0,
            resolution: 1024,
            crater_count: 1000,
        }
    }
}

impl LunarTerrain {
    pub fn generate(cfg: LunarTerrainConfig) -> Self {
        let LunarTerrainConfig { seed, size, resolution, crater_count } = cfg;
        let n = resolution.max(2);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut heights = vec![0.0_f32; n * n];

        // Layered value noise: a few octaves of bilerp'd random grids.
        let noise_layers: [(usize, f32); 3] = [(4, 1.4), (8, 0.55), (16, 0.22)];
        for (grid_size, amplitude) in noise_layers {
            let stride = grid_size + 1;
            let coarse: Vec<f32> = (0..stride * stride)
                .map(|_| rng.gen_range(-1.0_f32..1.0))
                .collect();
            for z in 0..n {
                for x in 0..n {
                    let fx = (x as f32 / (n - 1) as f32) * grid_size as f32;
                    let fz = (z as f32 / (n - 1) as f32) * grid_size as f32;
                    let x0 = (fx.floor() as usize).min(grid_size);
                    let z0 = (fz.floor() as usize).min(grid_size);
                    let x1 = (x0 + 1).min(grid_size);
                    let z1 = (z0 + 1).min(grid_size);
                    let mut tx = fx - x0 as f32;
                    let mut tz = fz - z0 as f32;
                    tx = tx * tx * (3.0 - 2.0 * tx);
                    tz = tz * tz * (3.0 - 2.0 * tz);
                    let v00 = coarse[z0 * stride + x0];
                    let v10 = coarse[z0 * stride + x1];
                    let v01 = coarse[z1 * stride + x0];
                    let v11 = coarse[z1 * stride + x1];
                    let v0 = v00 * (1.0 - tx) + v10 * tx;
                    let v1 = v01 * (1.0 - tx) + v11 * tx;
                    heights[z * n + x] += (v0 * (1.0 - tz) + v1 * tz) * amplitude;
                }
            }
        }

        // Stamp impact craters with parabolic bowls and gaussian rims.
        // Radii are uniform in [min_radius, max_radius]; depth is sampled
        // independently in an absolute range and is *not* tied to radius.
        // (Real impact craters have a similar property — large craters
        // are shallow relative to their width, small ones are deep
        // relative to their width.) Centres are scattered across the full
        // terrain square so coverage runs all the way to the mesh edges.
        let max_radius: f32 = 300.0;
        let min_radius: f32 = 18.0;
        for _ in 0..crater_count {
            let cx = rng.gen_range(-size..size);
            let cz = rng.gen_range(-size..size);
            let radius = rng.gen_range(min_radius..max_radius);
            let depth: f32 = rng.gen_range(2.0..15.0);
            stamp_crater(&mut heights, n, size, cx, cz, radius, depth);
        }

        // Flatten a landing pad around the origin BEFORE stamping the big
        // arena crater. This way the multiply-by-blend wipes the local
        // noise + small craters to zero, and the arena crater then dishes
        // the whole region (pad included) down to its bowl floor — so the
        // pad ends up as a small flat plateau referenced to the bottom of
        // the giant crater, not a pillar at the original sea level.
        let pad_radius = 30.0;
        let pad_falloff = 100.0;
        for z in 0..n {
            let wz = (z as f32 / (n - 1) as f32) * 2.0 * size - size;
            for x in 0..n {
                let wx = (x as f32 / (n - 1) as f32) * 2.0 * size - size;
                let d = (wx * wx + wz * wz).sqrt();
                if d < pad_radius + pad_falloff {
                    let t = ((d - pad_radius) / pad_falloff).clamp(0.0, 1.0);
                    let blend = t * t * (3.0 - 2.0 * t);
                    heights[z * n + x] *= blend;
                }
            }
        }

        // Arena-spanning fixed crater centred on the landing platform.
        // Uses the same `stamp_crater` math as the small random craters,
        // just with one fixed instance at origin. Radius = 0.8 × size,
        // depth = 0.1 × diameter.
        let arena_crater_radius = 0.8 * size;
        let arena_crater_depth = 0.075 * (2.0 * arena_crater_radius);
        stamp_crater(
            &mut heights,
            n,
            size,
            0.0,
            0.0,
            arena_crater_radius,
            arena_crater_depth,
        );

        Self { size, resolution: n, heights }
    }

    #[allow(dead_code)]
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let n = self.resolution;
        let u = (x + self.size) / (2.0 * self.size);
        let v = (z + self.size) / (2.0 * self.size);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return 0.0;
        }
        let fx = u * (n - 1) as f32;
        let fz = v * (n - 1) as f32;
        let x0 = (fx.floor() as usize).min(n - 1);
        let z0 = (fz.floor() as usize).min(n - 1);
        let x1 = (x0 + 1).min(n - 1);
        let z1 = (z0 + 1).min(n - 1);
        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;
        let h00 = self.heights[z0 * n + x0];
        let h10 = self.heights[z0 * n + x1];
        let h01 = self.heights[z1 * n + x0];
        let h11 = self.heights[z1 * n + x1];
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        h0 * (1.0 - tz) + h1 * tz
    }

    /// Build a Bevy mesh with per-vertex normals derived from the height gradient.
    pub fn build_mesh(&self) -> Mesh {
        let n = self.resolution;
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * n);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * n);
        let mut indices: Vec<u32> = Vec::with_capacity((n - 1) * (n - 1) * 6);

        for z in 0..n {
            for x in 0..n {
                let wx = (x as f32 / (n - 1) as f32) * 2.0 * self.size - self.size;
                let wz = (z as f32 / (n - 1) as f32) * 2.0 * self.size - self.size;
                let wy = self.heights[z * n + x];
                positions.push([wx, wy, wz]);
                uvs.push([x as f32 / (n - 1) as f32, z as f32 / (n - 1) as f32]);
            }
        }

        for z in 0..(n - 1) {
            for x in 0..(n - 1) {
                let i0 = (z * n + x) as u32;
                let i1 = (z * n + x + 1) as u32;
                let i2 = ((z + 1) * n + x) as u32;
                let i3 = ((z + 1) * n + x + 1) as u32;
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }

        let cell = 2.0 * self.size / (n - 1) as f32;
        let mut normals = vec![[0.0_f32, 1.0, 0.0]; n * n];
        for z in 0..n {
            for x in 0..n {
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(n - 1);
                let zl = z.saturating_sub(1);
                let zr = (z + 1).min(n - 1);
                let hl = self.heights[z * n + xl];
                let hr = self.heights[z * n + xr];
                let hd = self.heights[zl * n + x];
                let hu = self.heights[zr * n + x];
                let dx = (hr - hl) / ((xr - xl).max(1) as f32 * cell);
                let dz = (hu - hd) / ((zr - zl).max(1) as f32 * cell);
                let nv = Vec3::new(-dx, 1.0, -dz).normalize();
                normals[z * n + x] = [nv.x, nv.y, nv.z];
            }
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }
}


fn stamp_crater(
    heights: &mut [f32],
    n: usize,
    size: f32,
    cx: f32,
    cz: f32,
    radius: f32,
    depth: f32,
) {
    let influence = radius * 1.8;
    let rim_sigma = 0.22;
    let rim_height = depth * 0.45;

    for z in 0..n {
        let wz = (z as f32 / (n - 1) as f32) * 2.0 * size - size;
        let dz = wz - cz;
        if dz.abs() > influence { continue; }
        for x in 0..n {
            let wx = (x as f32 / (n - 1) as f32) * 2.0 * size - size;
            let dx = wx - cx;
            let dist = (dx * dx + dz * dz).sqrt();
            let d = dist / radius;
            if d > 1.8 { continue; }
            let bowl = if d < 1.0 {
                -depth * (1.0 - d * d)
            } else {
                0.0
            };
            let rim = rim_height
                * (-((d - 1.0).powi(2)) / (rim_sigma * rim_sigma)).exp();
            heights[z * n + x] += bowl + rim;
        }
    }
}
