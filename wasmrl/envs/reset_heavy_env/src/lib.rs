// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Reset-Heavy Environment for benchmarking fast reset optimization.
//!
//! This environment is designed to stress-test reset operations by having:
//! - Large initial state (configurable size)
//! - Short episodes (configurable max steps)
//! - Expensive initialization
//!
//! The environment simulates a grid world with obstacles and a goal.
//! The state includes the full grid (which can be large) plus agent position.
//!
//! # Configuration
//!
//! ```json
//! {
//!     "grid_size": 100,        // Grid is grid_size x grid_size
//!     "obstacle_density": 0.2, // Fraction of cells with obstacles
//!     "max_steps": 50,         // Episode truncates after this many steps
//!     "goal_x": 99,            // Goal position
//!     "goal_y": 99
//! }
//! ```

use serde::{Deserialize, Serialize};
use wasmrl_sdk_rust::DeterministicRng;
use wasmrl_wit::{DType, EnvConfig, EnvHandle, SnapshotData, StepResult, Tensor};

/// Environment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetHeavyConfig {
    /// Grid dimension (grid is grid_size x grid_size).
    #[serde(default = "default_grid_size")]
    pub grid_size: usize,
    /// Fraction of cells with obstacles (0.0 to 1.0).
    #[serde(default = "default_obstacle_density")]
    pub obstacle_density: f64,
    /// Maximum steps before truncation.
    #[serde(default = "default_max_steps")]
    pub max_steps: u64,
    /// Goal x position.
    #[serde(default)]
    pub goal_x: Option<usize>,
    /// Goal y position.
    #[serde(default)]
    pub goal_y: Option<usize>,
}

fn default_grid_size() -> usize {
    100
}
fn default_obstacle_density() -> f64 {
    0.2
}
fn default_max_steps() -> u64 {
    50
}

impl Default for ResetHeavyConfig {
    fn default() -> Self {
        Self {
            grid_size: default_grid_size(),
            obstacle_density: default_obstacle_density(),
            max_steps: default_max_steps(),
            goal_x: None,
            goal_y: None,
        }
    }
}

/// Cell types in the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Cell {
    /// Empty cell (passable).
    Empty = 0,
    /// Obstacle (impassable).
    Obstacle = 1,
    /// Goal cell.
    Goal = 2,
    /// Agent position.
    Agent = 3,
}

/// Environment state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetHeavyState {
    /// The grid (flattened, row-major).
    pub grid: Vec<u8>,
    /// Grid dimension.
    pub grid_size: usize,
    /// Agent x position.
    pub agent_x: usize,
    /// Agent y position.
    pub agent_y: usize,
    /// Goal x position.
    pub goal_x: usize,
    /// Goal y position.
    pub goal_y: usize,
    /// Current step count.
    pub steps: u64,
    /// Maximum steps.
    pub max_steps: u64,
    /// Current seed.
    pub seed: u64,
    /// RNG state.
    pub rng_state: u64,
}

impl ResetHeavyState {
    /// Get cell at position.
    pub fn get_cell(&self, x: usize, y: usize) -> Cell {
        let idx = y * self.grid_size + x;
        match self.grid.get(idx).copied().unwrap_or(0) {
            0 => Cell::Empty,
            1 => Cell::Obstacle,
            2 => Cell::Goal,
            3 => Cell::Agent,
            _ => Cell::Empty,
        }
    }

    /// Set cell at position.
    pub fn set_cell(&mut self, x: usize, y: usize, cell: Cell) {
        let idx = y * self.grid_size + x;
        if idx < self.grid.len() {
            self.grid[idx] = cell as u8;
        }
    }

    /// Check if position is valid and passable.
    pub fn is_passable(&self, x: usize, y: usize) -> bool {
        if x >= self.grid_size || y >= self.grid_size {
            return false;
        }
        let cell = self.get_cell(x, y);
        cell != Cell::Obstacle
    }

    /// Calculate distance to goal.
    pub fn distance_to_goal(&self) -> f64 {
        let dx = (self.agent_x as f64 - self.goal_x as f64).abs();
        let dy = (self.agent_y as f64 - self.goal_y as f64).abs();
        (dx * dx + dy * dy).sqrt()
    }

    /// Check if agent reached goal.
    pub fn at_goal(&self) -> bool {
        self.agent_x == self.goal_x && self.agent_y == self.goal_y
    }
}

/// Reset-Heavy Environment.
///
/// Actions:
/// - 0: Move up (y - 1)
/// - 1: Move down (y + 1)
/// - 2: Move left (x - 1)
/// - 3: Move right (x + 1)
/// - 4: No-op
///
/// Observation:
/// - Flattened grid + agent position + goal position
///
/// Rewards:
/// - +10.0 for reaching goal
/// - -0.1 per step (encourage efficiency)
/// - -1.0 for hitting obstacle (doesn't move)
pub struct ResetHeavyEnv {
    /// Configuration.
    config: ResetHeavyConfig,
    /// Current state.
    state: Option<ResetHeavyState>,
    /// Environment handle.
    handle: EnvHandle,
}

impl ResetHeavyEnv {
    /// Create a new environment.
    pub fn new(config: ResetHeavyConfig) -> Self {
        Self {
            config,
            state: None,
            handle: EnvHandle::new(0),
        }
    }

    /// Initialize from EnvConfig.
    pub fn init(env_config: &EnvConfig) -> anyhow::Result<Self> {
        let config: ResetHeavyConfig =
            if env_config.config_json.is_empty() || env_config.config_json == "{}" {
                ResetHeavyConfig::default()
            } else {
                serde_json::from_str(&env_config.config_json)?
            };

        Ok(Self::new(config))
    }

    /// Reset the environment with given seed.
    pub fn reset(&mut self, seed: u64) -> Tensor {
        let mut rng = DeterministicRng::new(seed);
        let size = self.config.grid_size;
        let total_cells = size * size;

        // Initialize grid with obstacles
        let mut grid = vec![Cell::Empty as u8; total_cells];
        let num_obstacles = (total_cells as f64 * self.config.obstacle_density) as usize;

        for _ in 0..num_obstacles {
            let idx = (rng.next_u64() as usize) % total_cells;
            grid[idx] = Cell::Obstacle as u8;
        }

        // Place agent at random empty cell (not at goal)
        let (agent_x, agent_y) = loop {
            let x = (rng.next_u64() as usize) % size;
            let y = (rng.next_u64() as usize) % size;
            let idx = y * size + x;
            if grid[idx] == Cell::Empty as u8 {
                break (x, y);
            }
        };

        // Place goal (ensure it's not on an obstacle or agent start)
        let goal_x = self.config.goal_x.unwrap_or(size - 1);
        let goal_y = self.config.goal_y.unwrap_or(size - 1);
        let goal_idx = goal_y * size + goal_x;
        grid[goal_idx] = Cell::Goal as u8;

        // Mark agent position
        let agent_idx = agent_y * size + agent_x;
        grid[agent_idx] = Cell::Agent as u8;

        let state = ResetHeavyState {
            grid,
            grid_size: size,
            agent_x,
            agent_y,
            goal_x,
            goal_y,
            steps: 0,
            max_steps: self.config.max_steps,
            seed,
            rng_state: rng.state(),
        };

        self.state = Some(state);
        self.get_observation()
    }

    /// Execute one step.
    pub fn step(&mut self, action: &Tensor) -> anyhow::Result<StepResult> {
        let (reward, terminated, truncated, steps, distance) = {
            let state = self
                .state
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Environment not reset"))?;

            // Decode action
            let action_value = if action.data.len() >= 4 {
                i32::from_le_bytes([
                    action.data[0],
                    action.data[1],
                    action.data[2],
                    action.data[3],
                ])
            } else {
                0
            };

            // Calculate new position
            let (new_x, new_y) = match action_value {
                0 => (state.agent_x, state.agent_y.saturating_sub(1)), // Up
                1 => (state.agent_x, (state.agent_y + 1).min(state.grid_size - 1)), // Down
                2 => (state.agent_x.saturating_sub(1), state.agent_y), // Left
                3 => ((state.agent_x + 1).min(state.grid_size - 1), state.agent_y), // Right
                _ => (state.agent_x, state.agent_y),                   // No-op
            };

            // Check if move is valid
            let mut reward = -0.1; // Step penalty

            if state.is_passable(new_x, new_y) {
                // Clear old position
                state.set_cell(state.agent_x, state.agent_y, Cell::Empty);

                // Update position
                state.agent_x = new_x;
                state.agent_y = new_y;

                // Mark new position (unless it's the goal)
                if !state.at_goal() {
                    state.set_cell(new_x, new_y, Cell::Agent);
                }
            } else {
                // Hit obstacle
                reward = -1.0;
            }

            state.steps += 1;

            // Check termination
            let terminated = state.at_goal();
            let truncated = state.steps >= state.max_steps;

            if terminated {
                reward = 10.0;
            }

            (
                reward,
                terminated,
                truncated,
                state.steps,
                state.distance_to_goal(),
            )
        };

        let obs = self.get_observation();

        Ok(StepResult {
            observation: obs,
            reward,
            terminated,
            truncated,
            info: Some(format!(
                "{{\"steps\": {}, \"distance\": {:.2}}}",
                steps, distance
            )),
        })
    }

    /// Get current observation.
    fn get_observation(&self) -> Tensor {
        let state = self.state.as_ref().unwrap();

        // Observation: grid + [agent_x, agent_y, goal_x, goal_y]
        let grid_size = state.grid_size;
        let obs_size = grid_size * grid_size + 4;
        let mut data = Vec::with_capacity(obs_size * 4);

        // Grid as f32
        for &cell in &state.grid {
            data.extend_from_slice(&(cell as f32).to_le_bytes());
        }

        // Agent and goal positions
        data.extend_from_slice(&(state.agent_x as f32).to_le_bytes());
        data.extend_from_slice(&(state.agent_y as f32).to_le_bytes());
        data.extend_from_slice(&(state.goal_x as f32).to_le_bytes());
        data.extend_from_slice(&(state.goal_y as f32).to_le_bytes());

        Tensor::new(DType::Float32, vec![obs_size as u32], data)
    }

    /// Take a snapshot of current state.
    pub fn snapshot(&self) -> anyhow::Result<SnapshotData> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No state to snapshot"))?;

        let data = serde_json::to_vec(state)?;
        Ok(SnapshotData::new(data))
    }

    /// Restore from a snapshot.
    pub fn restore(&mut self, snapshot: &SnapshotData) -> anyhow::Result<()> {
        if !snapshot.is_compatible() {
            anyhow::bail!("Incompatible snapshot version");
        }

        let state: ResetHeavyState = serde_json::from_slice(&snapshot.data)?;
        self.state = Some(state);
        Ok(())
    }

    /// Get state size in bytes (for benchmarking).
    pub fn state_size(&self) -> usize {
        self.state
            .as_ref()
            .map(|s| s.grid.len() + 64) // grid + other fields
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_env(grid_size: usize) -> ResetHeavyEnv {
        let config = ResetHeavyConfig {
            grid_size,
            obstacle_density: 0.1,
            max_steps: 100,
            goal_x: Some(grid_size - 1),
            goal_y: Some(grid_size - 1),
        };
        ResetHeavyEnv::new(config)
    }

    fn make_action(value: i32) -> Tensor {
        Tensor::new(DType::Int32, vec![1], value.to_le_bytes().to_vec())
    }

    #[test]
    fn test_env_creation() {
        let env = create_env(10);
        assert!(env.state.is_none());
    }

    #[test]
    fn test_env_reset() {
        let mut env = create_env(10);
        let obs = env.reset(42);

        assert!(env.state.is_some());
        let state = env.state.as_ref().unwrap();
        assert_eq!(state.grid_size, 10);
        assert_eq!(state.steps, 0);
        assert_eq!(obs.shape, vec![104]); // 10*10 + 4
    }

    #[test]
    fn test_env_step() {
        let mut env = create_env(10);
        env.reset(42);

        let result = env.step(&make_action(1)).unwrap(); // Down
        assert!(!result.terminated);
        assert!(!result.truncated);

        let state = env.state.as_ref().unwrap();
        assert_eq!(state.steps, 1);
    }

    #[test]
    fn test_env_truncation() {
        let config = ResetHeavyConfig {
            grid_size: 5,
            obstacle_density: 0.0, // No obstacles
            max_steps: 3,
            goal_x: Some(4),
            goal_y: Some(4),
        };
        let mut env = ResetHeavyEnv::new(config);
        env.reset(42);

        // Take 3 steps
        for _ in 0..3 {
            let result = env.step(&make_action(4)).unwrap(); // No-op
            if result.truncated {
                return;
            }
        }

        // Should be truncated by now
        let state = env.state.as_ref().unwrap();
        assert!(state.steps >= 3);
    }

    #[test]
    fn test_env_snapshot_restore() {
        let mut env = create_env(10);
        env.reset(42);

        // Take some steps
        for _ in 0..5 {
            env.step(&make_action(1)).unwrap();
        }

        let snapshot = env.snapshot().unwrap();
        let state_before = env.state.clone().unwrap();

        // Take more steps
        for _ in 0..5 {
            env.step(&make_action(1)).unwrap();
        }

        // Restore
        env.restore(&snapshot).unwrap();
        let state_after = env.state.as_ref().unwrap();

        assert_eq!(state_before.steps, state_after.steps);
        assert_eq!(state_before.agent_x, state_after.agent_x);
        assert_eq!(state_before.agent_y, state_after.agent_y);
    }

    #[test]
    fn test_env_determinism() {
        let mut env1 = create_env(20);
        let mut env2 = create_env(20);

        env1.reset(12345);
        env2.reset(12345);

        // Same seed should produce same state
        let state1 = env1.state.as_ref().unwrap();
        let state2 = env2.state.as_ref().unwrap();
        assert_eq!(state1.agent_x, state2.agent_x);
        assert_eq!(state1.agent_y, state2.agent_y);
        assert_eq!(state1.grid, state2.grid);
    }

    #[test]
    fn test_large_state_size() {
        let mut env = create_env(100);
        env.reset(42);

        // 100x100 grid = 10,000 cells
        let size = env.state_size();
        assert!(size >= 10000, "state size = {}", size);
    }

    #[test]
    fn test_cell_types() {
        assert_eq!(Cell::Empty as u8, 0);
        assert_eq!(Cell::Obstacle as u8, 1);
        assert_eq!(Cell::Goal as u8, 2);
        assert_eq!(Cell::Agent as u8, 3);
    }

    #[test]
    fn test_state_distance_to_goal() {
        let mut env = create_env(10);
        env.reset(42);

        let state = env.state.as_ref().unwrap();
        let dist = state.distance_to_goal();
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_init_from_env_config() {
        let env_config = EnvConfig::new(r#"{"grid_size": 50, "max_steps": 25}"#);
        let env = ResetHeavyEnv::init(&env_config).unwrap();

        assert_eq!(env.config.grid_size, 50);
        assert_eq!(env.config.max_steps, 25);
    }

    #[test]
    fn test_init_default_config() {
        let env_config = EnvConfig::empty();
        let env = ResetHeavyEnv::init(&env_config).unwrap();

        assert_eq!(env.config.grid_size, 100);
        assert_eq!(env.config.max_steps, 50);
    }
}
