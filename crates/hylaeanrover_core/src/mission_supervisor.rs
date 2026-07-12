//! Shared mission-level safety supervisor for RL and the in-game autopilot.
//!
//! PPO remains responsible for mineral exploration while power and attitude
//! are healthy. At low power, this state machine switches to a deterministic
//! cube intercept only when the nearest visible cube is reachable under the
//! real energy model; otherwise it preserves reserve by coasting. A tilt
//! guard has highest priority and removes throttle before a rollover develops.

use crate::minerals::element_catalog;
use crate::observation::{MAX_VISIBLE_CUBES, OBS_DIM};
use crate::reward::SCARCITY_WEIGHTS;
use crate::survey_coverage::COVERAGE_FEATURE_DIM;

const CUBE_OBS_START: usize = 15;
const CUBE_OBS_WIDTH: usize = 3;
const POWER_OBS_INDEX: usize = CUBE_OBS_START + MAX_VISIBLE_CUBES * CUBE_OBS_WIDTH;
const BEACONS_REMAINING_OBS_INDEX: usize = OBS_DIM - 1;
const MINERAL_OBS_START: usize = POWER_OBS_INDEX + 1;

pub const COAST_ACTION: u32 = 4;
pub const FORWARD_ACTION: u32 = 7;
pub const FORWARD_LEFT_ACTION: u32 = 6;
pub const FORWARD_RIGHT_ACTION: u32 = 8;

#[derive(Clone, Copy, Debug)]
pub struct MissionSupervisorConfig {
    pub low_power_enter_fraction: f32,
    pub low_power_exit_fraction: f32,
    pub path_safety_factor: f32,
    pub reserve_distance_m: f32,
    pub drain_wh_per_meter: f32,
    pub max_intercept_range_m: f32,
    pub recharge_detect_wh: f32,
    pub post_recharge_exploration_wh: f32,
    pub tilt_enter_deg: f32,
    pub tilt_exit_deg: f32,
    pub tilt_guard_min_speed_mps: f32,
    pub bearing_deadband_deg: f32,
    pub pickup_hold_range_m: f32,
    pub target_loss_grace_decisions: u32,
    pub beacon_guard_enabled: bool,
    pub beacon_first_distance_m: f32,
    pub beacon_spacing_m: f32,
    pub beacon_auto_deploy: bool,
    pub beacon_surface_score_threshold: f32,
}

impl Default for MissionSupervisorConfig {
    fn default() -> Self {
        Self {
            low_power_enter_fraction: 0.35,
            low_power_exit_fraction: 0.40,
            path_safety_factor: 1.10,
            reserve_distance_m: 2.0,
            drain_wh_per_meter: 0.5,
            max_intercept_range_m: 120.0,
            recharge_detect_wh: 50.0,
            post_recharge_exploration_wh: 75.0,
            tilt_enter_deg: 20.0,
            tilt_exit_deg: 18.0,
            tilt_guard_min_speed_mps: 1.0,
            bearing_deadband_deg: 6.0,
            pickup_hold_range_m: 3.0,
            target_loss_grace_decisions: 0,
            beacon_guard_enabled: true,
            beacon_first_distance_m: 100.0,
            beacon_spacing_m: 75.0,
            beacon_auto_deploy: true,
            beacon_surface_score_threshold: 150.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisorMode {
    Explore,
    Intercept,
    Commit,
    Preserve,
    Stabilize,
    BeaconDeploy,
    BeaconHold,
}

impl SupervisorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Intercept => "intercept",
            Self::Commit => "commit",
            Self::Preserve => "preserve",
            Self::Stabilize => "stabilize",
            Self::BeaconDeploy => "beacon_deploy",
            Self::BeaconHold => "beacon_hold",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SupervisorDecision {
    pub action: u32,
    pub proposed_action: u32,
    pub mode: SupervisorMode,
    pub overrode: bool,
    pub target_visible: bool,
    pub target_viable: bool,
    pub target_range_m: Option<f32>,
    pub available_range_m: f32,
}

#[derive(Clone, Debug)]
pub struct MissionSupervisor {
    config: MissionSupervisorConfig,
    recovering: bool,
    stabilizing: bool,
    target_committed: bool,
    target_loss_decisions: u32,
    last_beacon_distance_m: Option<f32>,
    last_power_wh: Option<f32>,
    recovery_rearm_wh: Option<f32>,
}

impl MissionSupervisor {
    pub fn new(config: MissionSupervisorConfig) -> Self {
        Self {
            config,
            recovering: false,
            stabilizing: false,
            target_committed: false,
            target_loss_decisions: 0,
            last_beacon_distance_m: None,
            last_power_wh: None,
            recovery_rearm_wh: None,
        }
    }

    pub fn config(&self) -> MissionSupervisorConfig {
        self.config
    }

    pub fn reset(&mut self) {
        self.recovering = false;
        self.stabilizing = false;
        self.target_committed = false;
        self.target_loss_decisions = 0;
        self.last_beacon_distance_m = None;
        self.last_power_wh = None;
        self.recovery_rearm_wh = None;
    }

    /// Return the observation presented to the mission policy.
    ///
    /// Cube interception and power management belong exclusively to this
    /// supervisor. Hiding cube slots and presenting a constant healthy battery
    /// prevents the mineral policy from chasing cubes or independently parking
    /// at its training-time reserve threshold. `decide` continues to use the
    /// unmodified observation for low-power recovery.
    pub fn policy_observation(&self, observation: &[f32]) -> Vec<f32> {
        self.policy_observation_with_coverage(observation, None)
    }

    pub fn policy_observation_with_coverage(
        &self,
        observation: &[f32],
        coverage_features: Option<&[f32; COVERAGE_FEATURE_DIM]>,
    ) -> Vec<f32> {
        let mut policy_observation = observation.to_vec();
        if policy_observation.len() > POWER_OBS_INDEX {
            policy_observation[CUBE_OBS_START..POWER_OBS_INDEX].fill(0.0);
            if let Some(features) = coverage_features {
                policy_observation[CUBE_OBS_START..POWER_OBS_INDEX].copy_from_slice(features);
            }
            policy_observation[POWER_OBS_INDEX] = 1.0;
        }
        policy_observation
    }

    pub fn decide(
        &mut self,
        observation: &[f32],
        proposed_action: u32,
        power_capacity_wh: f32,
    ) -> SupervisorDecision {
        self.decide_with_context(observation, proposed_action, power_capacity_wh, 0.0)
    }

    pub fn decide_with_context(
        &mut self,
        observation: &[f32],
        proposed_action: u32,
        power_capacity_wh: f32,
        distance_m: f32,
    ) -> SupervisorDecision {
        let proposed_action = if proposed_action < 10 {
            proposed_action
        } else {
            COAST_ACTION
        };
        if observation.len() < OBS_DIM {
            return self.decision(
                proposed_action,
                proposed_action,
                SupervisorMode::Explore,
                false,
                false,
                None,
                0.0,
            );
        }

        let power_fraction = observation[POWER_OBS_INDEX].clamp(0.0, 1.0);
        let power_capacity_wh = power_capacity_wh.max(0.0);
        let power_wh = power_fraction * power_capacity_wh;
        let power_gain_wh = self
            .last_power_wh
            .replace(power_wh)
            .map_or(0.0, |previous| power_wh - previous);
        let default_recovery_enter_wh = self.config.low_power_enter_fraction * power_capacity_wh;
        let recovery_enter_wh = self.recovery_rearm_wh.unwrap_or(default_recovery_enter_wh);
        if self.recovering {
            if power_fraction >= self.config.low_power_exit_fraction {
                self.recovering = false;
                self.recovery_rearm_wh = None;
                self.target_committed = false;
                self.target_loss_decisions = 0;
            } else if power_gain_wh >= self.config.recharge_detect_wh {
                self.recovering = false;
                self.recovery_rearm_wh = Some(
                    default_recovery_enter_wh
                        .min((power_wh - self.config.post_recharge_exploration_wh).max(0.0)),
                );
                self.target_committed = false;
                self.target_loss_decisions = 0;
            }
        } else if power_wh <= recovery_enter_wh {
            self.recovering = true;
            self.recovery_rearm_wh = None;
        }

        let speed_mps = observation[0].abs();
        let max_tilt = observation[2].abs().max(observation[3].abs());
        if self.stabilizing {
            if max_tilt <= self.config.tilt_exit_deg
                || speed_mps <= self.config.tilt_guard_min_speed_mps
            {
                self.stabilizing = false;
            }
        } else if max_tilt >= self.config.tilt_enter_deg
            && speed_mps > self.config.tilt_guard_min_speed_mps
        {
            self.stabilizing = true;
        }

        let available_range_m = if self.config.drain_wh_per_meter > 0.0 {
            power_wh / self.config.drain_wh_per_meter
        } else {
            0.0
        };
        let target = nearest_visible_cube(observation);

        if self.stabilizing {
            return self.decision(
                COAST_ACTION,
                proposed_action,
                SupervisorMode::Stabilize,
                target.is_some(),
                false,
                target.map(|(_, range)| range),
                available_range_m,
            );
        }

        if !self.recovering {
            if self.config.beacon_guard_enabled {
                let beacons_remaining = observation[BEACONS_REMAINING_OBS_INDEX];
                let required_distance_m = self
                    .last_beacon_distance_m
                    .map_or(self.config.beacon_first_distance_m, |last| {
                        last + self.config.beacon_spacing_m
                    });
                let placement_ready = beacons_remaining >= 0.5 && distance_m >= required_distance_m;
                let auto_deploy = self.config.beacon_auto_deploy
                    && beacon_surface_score(observation)
                        >= self.config.beacon_surface_score_threshold;
                if placement_ready && (auto_deploy || proposed_action == 9) {
                    self.last_beacon_distance_m = Some(distance_m);
                    return self.decision(
                        9,
                        proposed_action,
                        SupervisorMode::BeaconDeploy,
                        target.is_some(),
                        false,
                        target.map(|(_, range)| range),
                        available_range_m,
                    );
                }
                if proposed_action == 9 {
                    return self.decision(
                        COAST_ACTION,
                        proposed_action,
                        SupervisorMode::BeaconHold,
                        target.is_some(),
                        false,
                        target.map(|(_, range)| range),
                        available_range_m,
                    );
                }
            }
            return self.decision(
                proposed_action,
                proposed_action,
                SupervisorMode::Explore,
                target.is_some(),
                false,
                target.map(|(_, range)| range),
                available_range_m,
            );
        }

        if let Some((bearing, range)) = target {
            let required_range_m =
                range * self.config.path_safety_factor + self.config.reserve_distance_m;
            let viable =
                range <= self.config.max_intercept_range_m && required_range_m <= available_range_m;
            self.target_loss_decisions = 0;
            if viable {
                self.target_committed = true;
                let action = intercept_action(bearing, range, self.config);
                return self.decision(
                    action,
                    proposed_action,
                    SupervisorMode::Intercept,
                    true,
                    true,
                    Some(range),
                    available_range_m,
                );
            }

            self.target_committed = false;
            return self.decision(
                COAST_ACTION,
                proposed_action,
                SupervisorMode::Preserve,
                true,
                false,
                Some(range),
                available_range_m,
            );
        }

        if self.target_committed
            && self.target_loss_decisions < self.config.target_loss_grace_decisions
        {
            self.target_loss_decisions += 1;
            return self.decision(
                FORWARD_ACTION,
                proposed_action,
                SupervisorMode::Commit,
                false,
                true,
                None,
                available_range_m,
            );
        }

        self.target_committed = false;
        self.decision(
            COAST_ACTION,
            proposed_action,
            SupervisorMode::Preserve,
            false,
            false,
            None,
            available_range_m,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decision(
        &self,
        action: u32,
        proposed_action: u32,
        mode: SupervisorMode,
        target_visible: bool,
        target_viable: bool,
        target_range_m: Option<f32>,
        available_range_m: f32,
    ) -> SupervisorDecision {
        SupervisorDecision {
            action,
            proposed_action,
            mode,
            overrode: action != proposed_action,
            target_visible,
            target_viable,
            target_range_m,
            available_range_m,
        }
    }
}

fn nearest_visible_cube(observation: &[f32]) -> Option<(f32, f32)> {
    let mut nearest: Option<(f32, f32)> = None;
    for slot in 0..MAX_VISIBLE_CUBES {
        let start = CUBE_OBS_START + slot * CUBE_OBS_WIDTH;
        if observation[start + 2] <= 0.5 {
            continue;
        }
        let candidate = (observation[start], observation[start + 1]);
        if nearest.is_none_or(|(_, range)| candidate.1 < range) {
            nearest = Some(candidate);
        }
    }
    nearest
}

fn beacon_surface_score(observation: &[f32]) -> f32 {
    element_catalog()
        .zip(SCARCITY_WEIGHTS)
        .enumerate()
        .map(|(index, ((_name, base), weight))| {
            if base > 0.0 {
                weight * (observation[MINERAL_OBS_START + index].max(0.0) / base)
            } else {
                0.0
            }
        })
        .sum()
}

fn intercept_action(bearing: f32, range: f32, config: MissionSupervisorConfig) -> u32 {
    if range <= config.pickup_hold_range_m {
        COAST_ACTION
    } else if bearing.abs() <= config.bearing_deadband_deg {
        FORWARD_ACTION
    } else if bearing < 0.0 {
        FORWARD_RIGHT_ACTION
    } else {
        FORWARD_LEFT_ACTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(power: f32, pitch: f32, roll: f32, cube: Option<(f32, f32)>) -> Vec<f32> {
        let mut observation = vec![0.0; OBS_DIM];
        observation[0] = 2.0;
        observation[2] = pitch;
        observation[3] = roll;
        observation[POWER_OBS_INDEX] = power;
        if let Some((bearing, range)) = cube {
            observation[CUBE_OBS_START] = bearing;
            observation[CUBE_OBS_START + 1] = range;
            observation[CUBE_OBS_START + 2] = 1.0;
        }
        observation
    }

    #[test]
    fn healthy_power_preserves_policy_action() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let decision = supervisor.decide(&obs(0.9, 0.0, 0.0, None), 2, 100.0);
        assert_eq!(decision.action, 2);
        assert_eq!(decision.mode, SupervisorMode::Explore);
        assert!(!decision.overrode);
    }

    #[test]
    fn policy_observation_hides_supervisor_owned_power_and_cube_inputs() {
        let supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let observation = obs(0.20, 0.0, 0.0, Some((25.0, 40.0)));
        let policy_observation = supervisor.policy_observation(&observation);

        assert!(
            policy_observation[CUBE_OBS_START..POWER_OBS_INDEX]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_eq!(policy_observation[POWER_OBS_INDEX], 1.0);
        assert_eq!(observation[POWER_OBS_INDEX], 0.20);
        assert_eq!(nearest_visible_cube(&observation), Some((25.0, 40.0)));
    }

    #[test]
    fn low_power_without_target_preserves_reserve() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let decision = supervisor.decide(&obs(0.3, 0.0, 0.0, None), 7, 100.0);
        assert_eq!(decision.action, COAST_ACTION);
        assert_eq!(decision.mode, SupervisorMode::Preserve);
        assert!(decision.overrode);
    }

    #[test]
    fn low_power_reachable_target_uses_intercept_actions() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let decision = supervisor.decide(&obs(0.3, 0.0, 0.0, Some((12.0, 30.0))), 4, 100.0);
        assert_eq!(decision.action, FORWARD_LEFT_ACTION);
        assert_eq!(decision.mode, SupervisorMode::Intercept);
        assert!(decision.target_viable);
    }

    #[test]
    fn low_power_unreachable_target_preserves_reserve() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let decision = supervisor.decide(&obs(0.2, 0.0, 0.0, Some((0.0, 50.0))), 7, 100.0);
        assert_eq!(decision.action, COAST_ACTION);
        assert_eq!(decision.mode, SupervisorMode::Preserve);
        assert!(!decision.target_viable);
    }

    #[test]
    fn target_beyond_trained_intercept_range_preserves_reserve() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let decision = supervisor.decide(&obs(0.35, 0.0, 0.0, Some((0.0, 400.0))), 7, 1000.0);
        assert_eq!(decision.action, COAST_ACTION);
        assert_eq!(decision.mode, SupervisorMode::Preserve);
        assert!(!decision.target_viable);
    }

    #[test]
    fn recovery_hysteresis_holds_until_exit_threshold() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        supervisor.decide(&obs(0.3, 0.0, 0.0, None), 7, 100.0);
        let still_recovering = supervisor.decide(&obs(0.39, 0.0, 0.0, None), 7, 100.0);
        assert_eq!(still_recovering.mode, SupervisorMode::Preserve);
        let recovered = supervisor.decide(&obs(0.8, 0.0, 0.0, None), 7, 100.0);
        assert_eq!(recovered.mode, SupervisorMode::Explore);
        assert_eq!(recovered.action, 7);
    }

    #[test]
    fn cube_recharge_funds_exploration_below_fixed_exit_threshold() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let entered = supervisor.decide(&obs(0.30, 0.0, 0.0, None), 7, 1000.0);
        assert_eq!(entered.mode, SupervisorMode::Preserve);

        let recharged = supervisor.decide(&obs(0.39, 0.0, 0.0, None), 7, 1000.0);
        assert_eq!(recharged.mode, SupervisorMode::Explore);
        assert_eq!(recharged.action, 7);

        let budget_remaining = supervisor.decide(&obs(0.32, 0.0, 0.0, None), 7, 1000.0);
        assert_eq!(budget_remaining.mode, SupervisorMode::Explore);
        let budget_spent = supervisor.decide(&obs(0.31, 0.0, 0.0, None), 7, 1000.0);
        assert_eq!(budget_spent.mode, SupervisorMode::Preserve);
    }

    #[test]
    fn tilt_guard_has_priority_and_hysteresis() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let tilted = supervisor.decide(&obs(0.9, 60.0, 0.0, None), 7, 100.0);
        assert_eq!(tilted.mode, SupervisorMode::Stabilize);
        assert_eq!(tilted.action, COAST_ACTION);
        let still_tilted = supervisor.decide(&obs(0.9, 40.0, 0.0, None), 7, 100.0);
        assert_eq!(still_tilted.mode, SupervisorMode::Stabilize);
        let stable = supervisor.decide(&obs(0.9, 10.0, 0.0, None), 7, 100.0);
        assert_eq!(stable.mode, SupervisorMode::Explore);
    }

    #[test]
    fn tilt_guard_releases_once_speed_is_low() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        supervisor.decide(&obs(0.9, 25.0, 0.0, None), 7, 100.0);
        let mut stopped = obs(0.9, 25.0, 0.0, None);
        stopped[0] = 0.5;
        let decision = supervisor.decide(&stopped, 7, 100.0);
        assert_eq!(decision.mode, SupervisorMode::Explore);
        assert_eq!(decision.action, 7);
    }

    #[test]
    fn beacon_guard_requires_exploration_and_spacing() {
        let config = MissionSupervisorConfig {
            beacon_auto_deploy: false,
            ..MissionSupervisorConfig::default()
        };
        let mut supervisor = MissionSupervisor::new(config);
        let mut observation = obs(0.9, 0.0, 0.0, None);
        observation[BEACONS_REMAINING_OBS_INDEX] = 5.0;

        let early = supervisor.decide_with_context(&observation, 9, 100.0, 50.0);
        assert_eq!(early.action, COAST_ACTION);
        assert_eq!(early.mode, SupervisorMode::BeaconHold);

        let first = supervisor.decide_with_context(&observation, 9, 100.0, 100.0);
        assert_eq!(first.action, 9);
        assert_eq!(first.mode, SupervisorMode::BeaconDeploy);

        let crowded = supervisor.decide_with_context(&observation, 9, 100.0, 150.0);
        assert_eq!(crowded.action, COAST_ACTION);
        let spaced = supervisor.decide_with_context(&observation, 9, 100.0, 175.0);
        assert_eq!(spaced.action, 9);
    }

    #[test]
    fn beacon_auto_deploy_uses_surface_score_after_exploration() {
        let mut supervisor = MissionSupervisor::new(MissionSupervisorConfig::default());
        let mut observation = obs(0.9, 0.0, 0.0, None);
        observation[BEACONS_REMAINING_OBS_INDEX] = 5.0;
        observation[MINERAL_OBS_START + 4] = 2_000.0;

        let decision = supervisor.decide_with_context(&observation, 7, 100.0, 100.0);
        assert_eq!(decision.action, 9);
        assert_eq!(decision.mode, SupervisorMode::BeaconDeploy);
    }
}
