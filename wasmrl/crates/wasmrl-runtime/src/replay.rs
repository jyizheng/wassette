// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Replay hook for debugging and reproducing environment behavior.
//!
//! This module provides functionality to:
//! - Save snapshots at regular intervals
//! - Record action sequences
//! - Replay from any saved checkpoint
//! - Reproduce bugs deterministically

use std::collections::VecDeque;
use std::time::Instant;

use wasmrl_wit::{SnapshotData, StepResult, Tensor};

use crate::error::{RuntimeError, RuntimeResult};
use crate::instance::InstanceHandle;

/// Configuration for replay recording.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Save snapshot every K steps (0 = disabled).
    pub snapshot_interval: u64,
    /// Maximum number of snapshots to retain.
    pub max_snapshots: usize,
    /// Whether to record actions.
    pub record_actions: bool,
    /// Maximum number of actions to retain.
    pub max_actions: usize,
    /// Whether to record observations.
    pub record_observations: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: 100,
            max_snapshots: 10,
            record_actions: true,
            max_actions: 10000,
            record_observations: false, // Can be expensive
        }
    }
}

impl ReplayConfig {
    /// Create config with snapshots disabled.
    pub fn actions_only() -> Self {
        Self {
            snapshot_interval: 0,
            record_actions: true,
            ..Default::default()
        }
    }

    /// Create config with frequent snapshots.
    pub fn frequent_snapshots(interval: u64) -> Self {
        Self {
            snapshot_interval: interval,
            ..Default::default()
        }
    }
}

/// A recorded checkpoint with snapshot and step number.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Step number when this checkpoint was created.
    pub step: u64,
    /// Snapshot data.
    pub snapshot: SnapshotData,
    /// Timestamp when checkpoint was created.
    pub timestamp: Instant,
}

impl Checkpoint {
    /// Create a new checkpoint.
    pub fn new(step: u64, snapshot: SnapshotData) -> Self {
        Self {
            step,
            snapshot,
            timestamp: Instant::now(),
        }
    }
}

/// A recorded action.
#[derive(Debug, Clone)]
pub struct RecordedAction {
    /// Step number.
    pub step: u64,
    /// The action taken.
    pub action: Tensor,
    /// Result of the action (optional).
    pub result: Option<RecordedResult>,
}

/// Recorded result of an action.
#[derive(Debug, Clone)]
pub struct RecordedResult {
    /// Reward received.
    pub reward: f64,
    /// Whether episode terminated.
    pub terminated: bool,
    /// Whether episode was truncated.
    pub truncated: bool,
    /// Observation (if recorded).
    pub observation: Option<Tensor>,
}

impl RecordedResult {
    /// Create from StepResult.
    pub fn from_step_result(result: &StepResult, include_obs: bool) -> Self {
        Self {
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            observation: if include_obs {
                Some(result.observation.clone())
            } else {
                None
            },
        }
    }
}

/// Replay recorder for a single environment instance.
#[derive(Debug)]
pub struct ReplayRecorder {
    /// Configuration.
    config: ReplayConfig,
    /// Instance handle.
    handle: InstanceHandle,
    /// Current step count.
    step_count: u64,
    /// Current episode number.
    episode: u64,
    /// Saved checkpoints.
    checkpoints: VecDeque<Checkpoint>,
    /// Recorded actions.
    actions: VecDeque<RecordedAction>,
    /// Initial seed for this episode.
    initial_seed: u64,
}

impl ReplayRecorder {
    /// Create a new replay recorder.
    pub fn new(handle: InstanceHandle, config: ReplayConfig) -> Self {
        Self {
            config,
            handle,
            step_count: 0,
            episode: 0,
            checkpoints: VecDeque::new(),
            actions: VecDeque::new(),
            initial_seed: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config(handle: InstanceHandle) -> Self {
        Self::new(handle, ReplayConfig::default())
    }

    /// Record the start of a new episode.
    pub fn record_reset(&mut self, seed: u64, initial_snapshot: Option<SnapshotData>) {
        self.step_count = 0;
        self.episode += 1;
        self.initial_seed = seed;
        self.actions.clear();

        // Save initial checkpoint if snapshot provided
        if let Some(snapshot) = initial_snapshot {
            self.checkpoints.clear();
            self.add_checkpoint(snapshot);
        }
    }

    /// Record an action and its result.
    pub fn record_step(&mut self, action: Tensor, result: Option<&StepResult>) {
        self.step_count += 1;

        if self.config.record_actions {
            let recorded_result = result.map(|r| {
                RecordedResult::from_step_result(r, self.config.record_observations)
            });

            let recorded = RecordedAction {
                step: self.step_count,
                action,
                result: recorded_result,
            };

            self.actions.push_back(recorded);

            // Trim if over capacity
            while self.actions.len() > self.config.max_actions {
                self.actions.pop_front();
            }
        }
    }

    /// Add a checkpoint snapshot.
    pub fn add_checkpoint(&mut self, snapshot: SnapshotData) {
        let checkpoint = Checkpoint::new(self.step_count, snapshot);
        self.checkpoints.push_back(checkpoint);

        // Trim if over capacity
        while self.checkpoints.len() > self.config.max_snapshots {
            self.checkpoints.pop_front();
        }
    }

    /// Check if we should save a checkpoint at current step.
    pub fn should_checkpoint(&self) -> bool {
        if self.config.snapshot_interval == 0 {
            return false;
        }
        self.step_count % self.config.snapshot_interval == 0
    }

    /// Get the nearest checkpoint before or at given step.
    pub fn get_checkpoint_before(&self, step: u64) -> Option<&Checkpoint> {
        self.checkpoints
            .iter()
            .filter(|c| c.step <= step)
            .max_by_key(|c| c.step)
    }

    /// Get the most recent checkpoint.
    pub fn latest_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoints.back()
    }

    /// Get actions from a checkpoint to current.
    pub fn get_actions_from(&self, from_step: u64) -> Vec<&RecordedAction> {
        self.actions
            .iter()
            .filter(|a| a.step > from_step)
            .collect()
    }

    /// Get all actions for current episode.
    pub fn all_actions(&self) -> impl Iterator<Item = &RecordedAction> {
        self.actions.iter()
    }

    /// Get replay data for reproducing from a step.
    pub fn get_replay_data(&self, target_step: u64) -> Option<ReplayData> {
        // Find best checkpoint
        let checkpoint = self.get_checkpoint_before(target_step)?;

        // Get actions from checkpoint to target
        let actions: Vec<Tensor> = self
            .actions
            .iter()
            .filter(|a| a.step > checkpoint.step && a.step <= target_step)
            .map(|a| a.action.clone())
            .collect();

        Some(ReplayData {
            checkpoint: checkpoint.clone(),
            actions,
            target_step,
            initial_seed: self.initial_seed,
        })
    }

    /// Get current step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get current episode number.
    pub fn episode(&self) -> u64 {
        self.episode
    }

    /// Get number of checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Get number of recorded actions.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Get the handle.
    pub fn handle(&self) -> InstanceHandle {
        self.handle
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.actions.clear();
        self.step_count = 0;
    }
}

/// Data needed to replay to a specific step.
#[derive(Debug, Clone)]
pub struct ReplayData {
    /// Checkpoint to restore from.
    pub checkpoint: Checkpoint,
    /// Actions to replay after checkpoint.
    pub actions: Vec<Tensor>,
    /// Target step to reach.
    pub target_step: u64,
    /// Initial seed for the episode.
    pub initial_seed: u64,
}

impl ReplayData {
    /// Get number of steps to replay.
    pub fn steps_to_replay(&self) -> usize {
        self.actions.len()
    }
}

/// Manager for multiple replay recorders.
#[derive(Debug, Default)]
pub struct ReplayManager {
    /// Recorders by instance handle.
    recorders: std::collections::HashMap<u64, ReplayRecorder>,
    /// Default configuration.
    default_config: ReplayConfig,
}

impl ReplayManager {
    /// Create a new replay manager.
    pub fn new(default_config: ReplayConfig) -> Self {
        Self {
            recorders: std::collections::HashMap::new(),
            default_config,
        }
    }

    /// Get or create a recorder for an instance.
    pub fn get_or_create(&mut self, handle: InstanceHandle) -> &mut ReplayRecorder {
        self.recorders
            .entry(handle.id)
            .or_insert_with(|| ReplayRecorder::new(handle, self.default_config.clone()))
    }

    /// Get a recorder for an instance.
    pub fn get(&self, handle: InstanceHandle) -> Option<&ReplayRecorder> {
        self.recorders.get(&handle.id)
    }

    /// Get mutable recorder.
    pub fn get_mut(&mut self, handle: InstanceHandle) -> Option<&mut ReplayRecorder> {
        self.recorders.get_mut(&handle.id)
    }

    /// Remove a recorder.
    pub fn remove(&mut self, handle: InstanceHandle) -> Option<ReplayRecorder> {
        self.recorders.remove(&handle.id)
    }

    /// Get number of recorders.
    pub fn len(&self) -> usize {
        self.recorders.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.recorders.is_empty()
    }

    /// Clear all recorders.
    pub fn clear(&mut self) {
        self.recorders.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmrl_wit::DType;

    fn make_action(value: i32) -> Tensor {
        Tensor::new(
            DType::Int32,
            vec![1],
            value.to_le_bytes().to_vec(),
        )
    }

    fn make_snapshot(step: u64) -> SnapshotData {
        SnapshotData::new(format!("snapshot_{}", step).into_bytes())
    }

    #[test]
    fn test_replay_config_default() {
        let config = ReplayConfig::default();
        assert_eq!(config.snapshot_interval, 100);
        assert!(config.record_actions);
        assert!(!config.record_observations);
    }

    #[test]
    fn test_replay_config_actions_only() {
        let config = ReplayConfig::actions_only();
        assert_eq!(config.snapshot_interval, 0);
        assert!(config.record_actions);
    }

    #[test]
    fn test_checkpoint_creation() {
        let snapshot = make_snapshot(50);
        let checkpoint = Checkpoint::new(50, snapshot);
        assert_eq!(checkpoint.step, 50);
    }

    #[test]
    fn test_recorder_basic() {
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::default_config(handle);

        assert_eq!(recorder.step_count(), 0);
        assert_eq!(recorder.episode(), 0);

        recorder.record_reset(42, Some(make_snapshot(0)));
        assert_eq!(recorder.episode(), 1);
        assert_eq!(recorder.checkpoint_count(), 1);
    }

    #[test]
    fn test_recorder_record_steps() {
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::default_config(handle);

        recorder.record_reset(42, None);

        for i in 0..10 {
            recorder.record_step(make_action(i), None);
        }

        assert_eq!(recorder.step_count(), 10);
        assert_eq!(recorder.action_count(), 10);
    }

    #[test]
    fn test_recorder_should_checkpoint() {
        let config = ReplayConfig {
            snapshot_interval: 5,
            ..Default::default()
        };
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::new(handle, config);

        recorder.record_reset(42, None);

        for i in 0..10 {
            recorder.record_step(make_action(i), None);
            if recorder.should_checkpoint() {
                recorder.add_checkpoint(make_snapshot(recorder.step_count()));
            }
        }

        // Should have checkpoints at steps 5 and 10
        assert_eq!(recorder.checkpoint_count(), 2);
    }

    #[test]
    fn test_recorder_get_checkpoint_before() {
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::default_config(handle);

        recorder.record_reset(42, Some(make_snapshot(0)));

        for i in 0..15 {
            recorder.record_step(make_action(i), None);
            if recorder.step_count() % 5 == 0 {
                recorder.add_checkpoint(make_snapshot(recorder.step_count()));
            }
        }

        // Checkpoints at 0, 5, 10, 15
        let cp = recorder.get_checkpoint_before(12).unwrap();
        assert_eq!(cp.step, 10);

        let cp = recorder.get_checkpoint_before(5).unwrap();
        assert_eq!(cp.step, 5);

        let cp = recorder.get_checkpoint_before(3).unwrap();
        assert_eq!(cp.step, 0);
    }

    #[test]
    fn test_recorder_get_actions_from() {
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::default_config(handle);

        recorder.record_reset(42, None);

        for i in 0..10 {
            recorder.record_step(make_action(i), None);
        }

        let actions = recorder.get_actions_from(5);
        assert_eq!(actions.len(), 5); // Steps 6-10
    }

    #[test]
    fn test_recorder_get_replay_data() {
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::default_config(handle);

        recorder.record_reset(42, Some(make_snapshot(0)));

        for i in 0..10 {
            recorder.record_step(make_action(i), None);
        }
        recorder.add_checkpoint(make_snapshot(10));

        for i in 10..15 {
            recorder.record_step(make_action(i), None);
        }

        // Get replay data to step 12
        let replay = recorder.get_replay_data(12).unwrap();
        assert_eq!(replay.checkpoint.step, 10);
        assert_eq!(replay.actions.len(), 2); // Steps 11-12
        assert_eq!(replay.target_step, 12);
        assert_eq!(replay.initial_seed, 42);
    }

    #[test]
    fn test_recorder_max_checkpoints() {
        let config = ReplayConfig {
            max_snapshots: 3,
            snapshot_interval: 1,
            ..Default::default()
        };
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::new(handle, config);

        recorder.record_reset(42, None);

        for i in 0..10 {
            recorder.record_step(make_action(i), None);
            recorder.add_checkpoint(make_snapshot(recorder.step_count()));
        }

        // Should only keep last 3
        assert_eq!(recorder.checkpoint_count(), 3);
        let latest = recorder.latest_checkpoint().unwrap();
        assert_eq!(latest.step, 10);
    }

    #[test]
    fn test_recorder_max_actions() {
        let config = ReplayConfig {
            max_actions: 5,
            ..Default::default()
        };
        let handle = InstanceHandle { id: 1 };
        let mut recorder = ReplayRecorder::new(handle, config);

        recorder.record_reset(42, None);

        for i in 0..10 {
            recorder.record_step(make_action(i), None);
        }

        // Should only keep last 5
        assert_eq!(recorder.action_count(), 5);
    }

    #[test]
    fn test_replay_manager() {
        let mut manager = ReplayManager::new(ReplayConfig::default());

        let h1 = InstanceHandle { id: 1 };
        let h2 = InstanceHandle { id: 2 };

        manager.get_or_create(h1).record_reset(42, None);
        manager.get_or_create(h2).record_reset(43, None);

        assert_eq!(manager.len(), 2);

        manager.get_mut(h1).unwrap().record_step(make_action(0), None);
        assert_eq!(manager.get(h1).unwrap().step_count(), 1);

        manager.remove(h1);
        assert_eq!(manager.len(), 1);
        assert!(manager.get(h1).is_none());
    }

    #[test]
    fn test_recorded_result_from_step_result() {
        let obs = Tensor::zeros(DType::Float32, vec![4]);
        let step_result = StepResult::new(obs.clone(), 1.5, false, false);

        let recorded = RecordedResult::from_step_result(&step_result, false);
        assert_eq!(recorded.reward, 1.5);
        assert!(!recorded.terminated);
        assert!(recorded.observation.is_none());

        let recorded_with_obs = RecordedResult::from_step_result(&step_result, true);
        assert!(recorded_with_obs.observation.is_some());
    }
}
