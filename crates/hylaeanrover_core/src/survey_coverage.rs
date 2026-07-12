//! Per-game survey memory shared by the rendered game and headless RL env.

use bevy::prelude::*;

use crate::ChassisEntity;
use crate::minerals::{MineralMaps, element_catalog};
use crate::power_cubes::RelaunchEvent;
use crate::reward::{MINERAL_INTEGRAL_SCALE, SCARCITY_WEIGHTS};

pub const COVERAGE_VERSION: u32 = 1;
pub const COVERAGE_CELL_SIZE_M: f32 = 5.0;
pub const COVERAGE_HALF_EXTENT_M: f32 = 2475.0;
pub const COVERAGE_FEATURE_DIM: usize = 18;
const SEGMENT_SAMPLE_SPACING_M: f32 = COVERAGE_CELL_SIZE_M * 0.5;
const TELEPORT_DISTANCE_M: f32 = 5.0;

#[derive(Resource, Clone, Debug)]
pub struct SurveyCoverage {
    visits: Vec<u8>,
    width: usize,
    previous_position: Option<Vec2>,
    previous_cell: Option<usize>,
    pub unique_cells: u32,
    pub revisit_entries: u32,
    pub novel_distance_m: f32,
    pub novel_mineral_integral: f32,
}

impl Default for SurveyCoverage {
    fn default() -> Self {
        let width = ((2.0 * COVERAGE_HALF_EXTENT_M) / COVERAGE_CELL_SIZE_M).ceil() as usize;
        Self {
            visits: vec![0; width * width],
            width,
            previous_position: None,
            previous_cell: None,
            unique_cells: 0,
            revisit_entries: 0,
            novel_distance_m: 0.0,
            novel_mineral_integral: 0.0,
        }
    }
}

impl SurveyCoverage {
    pub fn allocated_cells(&self) -> usize {
        self.visits.len()
    }

    pub fn covered_area_m2(&self) -> f32 {
        self.unique_cells as f32 * COVERAGE_CELL_SIZE_M * COVERAGE_CELL_SIZE_M
    }

    pub fn revisit_rate(&self) -> f32 {
        let entries = self.unique_cells.saturating_sub(1) + self.revisit_entries;
        if entries == 0 {
            0.0
        } else {
            self.revisit_entries as f32 / entries as f32
        }
    }

    pub fn reset(&mut self) {
        self.visits.fill(0);
        self.previous_position = None;
        self.previous_cell = None;
        self.unique_cells = 0;
        self.revisit_entries = 0;
        self.novel_distance_m = 0.0;
        self.novel_mineral_integral = 0.0;
    }

    pub fn cell_for_position(&self, position: Vec2) -> Option<(usize, usize)> {
        if position.x < -COVERAGE_HALF_EXTENT_M
            || position.x > COVERAGE_HALF_EXTENT_M
            || position.y < -COVERAGE_HALF_EXTENT_M
            || position.y > COVERAGE_HALF_EXTENT_M
        {
            return None;
        }
        let coordinate = |value: f32| {
            (((value + COVERAGE_HALF_EXTENT_M) / COVERAGE_CELL_SIZE_M).floor() as usize)
                .min(self.width - 1)
        };
        Some((coordinate(position.x), coordinate(position.y)))
    }

    /// Cells per grid side (the grid is square).
    pub fn grid_width(&self) -> usize {
        self.width
    }

    /// Visit count at grid cell (x, z); `None` outside the grid.
    pub fn visit_at_cell(&self, x: usize, z: usize) -> Option<u8> {
        (x < self.width && z < self.width).then(|| self.visits[self.index(x, z)])
    }

    pub fn visit_count_at(&self, position: Vec2) -> u8 {
        self.cell_for_position(position)
            .map(|(x, z)| self.visits[self.index(x, z)])
            .unwrap_or(u8::MAX)
    }

    pub fn mark_position<F>(&mut self, position: Vec2, mut mineral_score: F)
    where
        F: FnMut(Vec2) -> f32,
    {
        let Some(previous) = self.previous_position.replace(position) else {
            self.enter_position(position, false, &mut mineral_score);
            return;
        };
        let distance = previous.distance(position);
        if distance >= TELEPORT_DISTANCE_M {
            self.previous_cell = None;
            self.enter_position(position, false, &mut mineral_score);
            return;
        }
        if distance <= f32::EPSILON {
            return;
        }
        let steps = (distance / SEGMENT_SAMPLE_SPACING_M).ceil().max(1.0) as usize;
        for step in 1..=steps {
            let sample = previous.lerp(position, step as f32 / steps as f32);
            self.enter_position(sample, true, &mut mineral_score);
        }
    }

    pub fn frontier_features(&self, position: Vec2, heading_deg: f32) -> [f32; 18] {
        let mut features = [0.0; COVERAGE_FEATURE_DIM];
        let mut near_unvisited = [0_u32; 8];
        let mut near_total = [0_u32; 8];
        let mut far_unvisited = [0_u32; 8];
        let mut far_total = [0_u32; 8];
        let radius_cells = (50.0 / COVERAGE_CELL_SIZE_M).ceil() as i32;
        let heading = heading_deg.to_radians();
        let forward = Vec2::new(-heading.cos(), heading.sin());
        let mut local_unvisited = 0_u32;
        let mut local_total = 0_u32;

        let Some((center_x, center_z)) = self.cell_for_position(position) else {
            return features;
        };
        for dz in -radius_cells..=radius_cells {
            for dx in -radius_cells..=radius_cells {
                let x = center_x as i32 + dx;
                let z = center_z as i32 + dz;
                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.width as i32 {
                    continue;
                }
                let cell_position = self.cell_center(x as usize, z as usize);
                let offset = cell_position - position;
                let range = offset.length();
                if !(5.0..=50.0).contains(&range) {
                    continue;
                }
                let direction = offset / range;
                let cross = forward.x * direction.y - forward.y * direction.x;
                let dot = forward.dot(direction);
                let angle = cross.atan2(dot);
                let sector = (((angle + std::f32::consts::FRAC_PI_8)
                    .rem_euclid(std::f32::consts::TAU)
                    / std::f32::consts::FRAC_PI_4)
                    .floor() as usize)
                    % 8;
                let unvisited = self.visits[self.index(x as usize, z as usize)] == 0;
                local_total += 1;
                local_unvisited += u32::from(unvisited);
                if range <= 20.0 {
                    near_total[sector] += 1;
                    near_unvisited[sector] += u32::from(unvisited);
                } else {
                    far_total[sector] += 1;
                    far_unvisited[sector] += u32::from(unvisited);
                }
            }
        }
        for sector in 0..8 {
            features[sector] = fraction(near_unvisited[sector], near_total[sector]);
            features[8 + sector] = fraction(far_unvisited[sector], far_total[sector]);
        }
        features[16] = (self.visit_count_at(position).min(4) as f32) / 4.0;
        features[17] = fraction(local_unvisited, local_total);
        features
    }

    fn enter_position<F>(&mut self, position: Vec2, reward_new_cell: bool, mineral_score: &mut F)
    where
        F: FnMut(Vec2) -> f32,
    {
        let Some((x, z)) = self.cell_for_position(position) else {
            self.previous_cell = None;
            return;
        };
        let index = self.index(x, z);
        if self.previous_cell == Some(index) {
            return;
        }
        self.previous_cell = Some(index);
        let first_visit = self.visits[index] == 0;
        self.visits[index] = self.visits[index].saturating_add(1);
        if first_visit {
            self.unique_cells += 1;
            if reward_new_cell {
                self.novel_distance_m += COVERAGE_CELL_SIZE_M;
                self.novel_mineral_integral += mineral_score(self.cell_center(x, z))
                    * COVERAGE_CELL_SIZE_M
                    * MINERAL_INTEGRAL_SCALE;
            }
        } else {
            self.revisit_entries += 1;
        }
    }

    fn index(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    fn cell_center(&self, x: usize, z: usize) -> Vec2 {
        Vec2::new(
            -COVERAGE_HALF_EXTENT_M + (x as f32 + 0.5) * COVERAGE_CELL_SIZE_M,
            -COVERAGE_HALF_EXTENT_M + (z as f32 + 0.5) * COVERAGE_CELL_SIZE_M,
        )
    }
}

pub struct SurveyCoveragePlugin;

impl Plugin for SurveyCoveragePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SurveyCoverage>()
            .add_systems(Update, update_survey_coverage);
    }
}

fn update_survey_coverage(
    mut relaunches: MessageReader<RelaunchEvent>,
    chassis: Res<ChassisEntity>,
    transforms: Query<&GlobalTransform>,
    minerals: Res<MineralMaps>,
    mut coverage: ResMut<SurveyCoverage>,
) {
    if relaunches.read().next().is_some() {
        coverage.reset();
        return;
    }
    let Some(position) = chassis
        .0
        .and_then(|entity| transforms.get(entity).ok())
        .map(|transform| transform.translation())
    else {
        return;
    };
    coverage.mark_position(Vec2::new(position.x, position.z), |sample| {
        weighted_mineral_score(&minerals, sample)
    });
}

fn weighted_mineral_score(minerals: &MineralMaps, position: Vec2) -> f32 {
    minerals
        .surface_all_at(position.x, position.y)
        .into_iter()
        .zip(SCARCITY_WEIGHTS.iter().zip(element_catalog()))
        .map(|((_, value), (weight, (_, base)))| {
            if base > 0.0 {
                weight * (value / base)
            } else {
                0.0
            }
        })
        .sum()
}

fn fraction(numerator: u32, denominator: u32) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_grid_is_bounded_and_maps_edges() {
        let coverage = SurveyCoverage::default();
        assert_eq!(coverage.allocated_cells(), 990 * 990);
        assert_eq!(
            coverage.cell_for_position(Vec2::splat(-2475.0)),
            Some((0, 0))
        );
        assert_eq!(
            coverage.cell_for_position(Vec2::splat(2475.0)),
            Some((989, 989))
        );
        assert_eq!(coverage.cell_for_position(Vec2::new(2475.1, 0.0)), None);
    }

    #[test]
    fn cell_accessors_read_visits_and_reject_out_of_bounds() {
        let mut coverage = SurveyCoverage::default();
        assert_eq!(coverage.grid_width(), 990);
        assert_eq!(coverage.visit_at_cell(0, 0), Some(0));
        assert_eq!(coverage.visit_at_cell(990, 0), None);
        assert_eq!(coverage.visit_at_cell(0, 990), None);

        coverage.mark_position(Vec2::ZERO, |_| 1.0);
        let (x, z) = coverage.cell_for_position(Vec2::ZERO).unwrap();
        assert_eq!(coverage.visit_at_cell(x, z), Some(1));
    }

    #[test]
    fn stationary_updates_do_not_count_revisits() {
        let mut coverage = SurveyCoverage::default();
        coverage.mark_position(Vec2::ZERO, |_| 1.0);
        coverage.mark_position(Vec2::ZERO, |_| 1.0);
        assert_eq!(coverage.unique_cells, 1);
        assert_eq!(coverage.revisit_entries, 0);
        assert_eq!(coverage.novel_distance_m, 0.0);
    }

    #[test]
    fn segment_marks_new_cells_once_and_revisits_later() {
        let mut coverage = SurveyCoverage::default();
        coverage.mark_position(Vec2::ZERO, |_| 2.0);
        for x in 1..=12 {
            coverage.mark_position(Vec2::new(x as f32, 0.0), |_| 2.0);
        }
        let first_novel = coverage.novel_distance_m;
        assert!(first_novel >= 10.0);
        for x in (0..12).rev() {
            coverage.mark_position(Vec2::new(x as f32, 0.0), |_| 2.0);
        }
        assert_eq!(coverage.novel_distance_m, first_novel);
        assert!(coverage.revisit_entries > 0);
    }

    #[test]
    fn visits_saturate_and_reset_without_reallocation() {
        let mut coverage = SurveyCoverage::default();
        let allocated = coverage.allocated_cells();
        for _ in 0..300 {
            coverage.previous_position = None;
            coverage.previous_cell = None;
            coverage.mark_position(Vec2::ZERO, |_| 0.0);
        }
        assert_eq!(coverage.visit_count_at(Vec2::ZERO), u8::MAX);
        coverage.reset();
        assert_eq!(coverage.allocated_cells(), allocated);
        assert_eq!(coverage.visit_count_at(Vec2::ZERO), 0);
    }

    #[test]
    fn frontier_features_rotate_visited_ground_behind_rover() {
        let mut coverage = SurveyCoverage::default();
        for x in 0..=40 {
            coverage.mark_position(Vec2::new(-(x as f32), 0.0), |_| 0.0);
        }
        let at_end = Vec2::new(-40.0, 0.0);
        let facing_forward = coverage.frontier_features(at_end, 0.0);
        let facing_back = coverage.frontier_features(at_end, 180.0);
        assert!(facing_forward[4] < facing_forward[0]);
        assert!(facing_back[0] < facing_back[4]);
    }
}
