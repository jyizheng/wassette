// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Fast reset using snapshot restore.
//!
//! This module provides optimized reset operations that use cached snapshots
//! instead of re-initializing environments from scratch.

use std::sync::Arc;
use std::time::Duration;

use wasmrl_wit::{EnvConfig, SnapshotData, Tensor};

use crate::error::{RuntimeError, RuntimeResult};
use crate::instance::InstanceHandle;
use crate::metrics::{LatencyStats, Timer};
use crate::snapshot_cache::{SharedSnapshotCache, SnapshotKey};

/// Configuration for fast reset behavior.
#[derive(Debug, Clone)]
pub struct FastResetConfig {
    /// Whether to enable fast reset.
    pub enabled: bool,
    /// Whether to automatically cache initial snapshots.
    pub auto_cache: bool,
    /// Maximum snapshot size to cache (bytes).
    pub max_snapshot_size: usize,
    /// Cache capacity (number of entries).
    pub cache_capacity: usize,
    /// Maximum cache size in bytes.
    pub cache_max_bytes: usize,
}

impl Default for FastResetConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_cache: true,
            max_snapshot_size: 10 * 1024 * 1024, // 10 MB
            cache_capacity: 1000,
            cache_max_bytes: 100 * 1024 * 1024, // 100 MB
        }
    }
}

impl FastResetConfig {
    /// Create config with fast reset disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create config with custom cache size.
    pub fn with_cache(capacity: usize, max_bytes: usize) -> Self {
        Self {
            cache_capacity: capacity,
            cache_max_bytes: max_bytes,
            ..Default::default()
        }
    }
}

/// Manager for fast reset operations.
#[derive(Debug)]
pub struct FastResetManager {
    /// Configuration.
    config: FastResetConfig,
    /// Snapshot cache.
    cache: SharedSnapshotCache,
    /// Metrics for fast reset.
    metrics: FastResetMetrics,
}

/// Metrics specific to fast reset operations.
#[derive(Debug, Default)]
pub struct FastResetMetrics {
    /// Full reset latency (without cache).
    pub full_reset_latency: LatencyStats,
    /// Fast reset latency (with cache).
    pub fast_reset_latency: LatencyStats,
    /// Number of full resets performed.
    pub full_resets: u64,
    /// Number of fast resets performed.
    pub fast_resets: u64,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Snapshots cached.
    pub snapshots_cached: u64,
}

impl FastResetMetrics {
    /// Create new metrics.
    pub fn new() -> Self {
        Self {
            full_reset_latency: LatencyStats::new(1000),
            fast_reset_latency: LatencyStats::new(1000),
            ..Default::default()
        }
    }

    /// Calculate speedup ratio (full / fast).
    pub fn speedup_ratio(&self) -> f64 {
        let full_mean = self.full_reset_latency.mean_ns();
        let fast_mean = self.fast_reset_latency.mean_ns();
        if fast_mean == 0 {
            0.0
        } else {
            full_mean as f64 / fast_mean as f64
        }
    }

    /// Get fast reset rate.
    pub fn fast_reset_rate(&self) -> f64 {
        let total = self.full_resets + self.fast_resets;
        if total == 0 {
            0.0
        } else {
            self.fast_resets as f64 / total as f64
        }
    }

    /// Average full reset latency.
    pub fn avg_full_reset_time(&self) -> Duration {
        self.full_reset_latency.mean()
    }

    /// Average cached reset latency.
    pub fn avg_fast_reset_time(&self) -> Duration {
        self.fast_reset_latency.mean()
    }
}

impl FastResetManager {
    /// Create a new fast reset manager.
    pub fn new(config: FastResetConfig) -> Self {
        let cache =
            SharedSnapshotCache::with_byte_limit(config.cache_capacity, config.cache_max_bytes);

        Self {
            config,
            cache,
            metrics: FastResetMetrics::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(FastResetConfig::default())
    }

    /// Check if fast reset is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get or create a snapshot key for given parameters.
    pub fn make_key(&self, component_id: &str, config: &EnvConfig, seed: u64) -> SnapshotKey {
        SnapshotKey::new(component_id, &config.config_json, seed)
    }

    /// Try to get a cached snapshot for fast reset.
    pub fn get_cached(&mut self, key: &SnapshotKey) -> Option<SnapshotData> {
        if !self.config.enabled {
            return None;
        }

        match self.cache.get(key) {
            Some(snapshot) => {
                self.metrics.cache_hits += 1;
                Some(snapshot)
            }
            None => {
                self.metrics.cache_misses += 1;
                None
            }
        }
    }

    /// Cache a snapshot for future fast resets.
    pub fn cache_snapshot(&mut self, key: SnapshotKey, snapshot: SnapshotData) {
        if !self.config.enabled {
            return;
        }

        // Check size limit
        if snapshot.data.len() > self.config.max_snapshot_size {
            return;
        }

        self.cache.put(key, snapshot);
        self.metrics.snapshots_cached += 1;
    }

    /// Cache raw initial state bytes for future fast resets.
    pub fn cache_initial_state(&mut self, key: &SnapshotKey, state: Vec<u8>) {
        self.cache_snapshot(key.clone(), SnapshotData::new(state));
    }

    /// Get raw cached state bytes for a fast reset.
    pub fn get_cached_state(&mut self, key: &SnapshotKey) -> Option<Vec<u8>> {
        self.get_cached(key).map(|snapshot| snapshot.data)
    }

    /// Check if a snapshot is cached.
    pub fn is_cached(&self, key: &SnapshotKey) -> bool {
        self.cache.contains(key)
    }

    /// Record a full reset operation.
    pub fn record_full_reset(&mut self, duration: Duration) {
        self.metrics.full_reset_latency.record(duration);
        self.metrics.full_resets += 1;
    }

    /// Record a fast reset operation.
    pub fn record_fast_reset(&mut self, duration: Duration) {
        self.metrics.fast_reset_latency.record(duration);
        self.metrics.fast_resets += 1;
    }

    /// Get metrics.
    pub fn metrics(&self) -> &FastResetMetrics {
        &self.metrics
    }

    /// Get cache reference.
    pub fn cache(&self) -> &SharedSnapshotCache {
        &self.cache
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get configuration.
    pub fn config(&self) -> &FastResetConfig {
        &self.config
    }
}

/// Result of a reset operation with timing info.
#[derive(Debug)]
pub struct ResetResult {
    /// Initial observation.
    pub observation: Tensor,
    /// Whether fast reset was used.
    pub was_fast: bool,
    /// Duration of the reset operation.
    pub duration: Duration,
}

impl ResetResult {
    /// Create a new reset result.
    pub fn new(observation: Tensor, was_fast: bool, duration: Duration) -> Self {
        Self {
            observation,
            was_fast,
            duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use wasmrl_wit::DType;

    use super::*;

    #[test]
    fn test_fast_reset_config_default() {
        let config = FastResetConfig::default();
        assert!(config.enabled);
        assert!(config.auto_cache);
        assert_eq!(config.cache_capacity, 1000);
    }

    #[test]
    fn test_fast_reset_config_disabled() {
        let config = FastResetConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_fast_reset_manager_creation() {
        let manager = FastResetManager::default_config();
        assert!(manager.is_enabled());
    }

    #[test]
    fn test_make_key() {
        let manager = FastResetManager::default_config();
        let config = EnvConfig::new("{\"target\": 10}");
        let key = manager.make_key("counter_env", &config, 42);

        assert_eq!(key.component_id, "counter_env");
        assert_eq!(key.config_hash, "{\"target\": 10}");
        assert_eq!(key.seed, 42);
    }

    #[test]
    fn test_cache_and_retrieve() {
        let mut manager = FastResetManager::default_config();
        let key = SnapshotKey::new("env", "cfg", 42);
        let snapshot = SnapshotData::new(vec![1, 2, 3, 4]);

        // Cache miss first
        assert!(manager.get_cached(&key).is_none());
        assert_eq!(manager.metrics.cache_misses, 1);

        // Cache the snapshot
        manager.cache_snapshot(key.clone(), snapshot.clone());
        assert_eq!(manager.metrics.snapshots_cached, 1);

        // Cache hit
        let retrieved = manager.get_cached(&key).unwrap();
        assert_eq!(retrieved.data, vec![1, 2, 3, 4]);
        assert_eq!(manager.metrics.cache_hits, 1);
    }

    #[test]
    fn test_disabled_manager() {
        let mut manager = FastResetManager::new(FastResetConfig::disabled());
        let key = SnapshotKey::new("env", "cfg", 42);
        let snapshot = SnapshotData::new(vec![1, 2, 3, 4]);

        // Should not cache when disabled
        manager.cache_snapshot(key.clone(), snapshot);
        assert!(manager.get_cached(&key).is_none());
    }

    #[test]
    fn test_size_limit() {
        let config = FastResetConfig {
            max_snapshot_size: 100,
            ..Default::default()
        };
        let mut manager = FastResetManager::new(config);

        let key = SnapshotKey::new("env", "cfg", 42);
        let large_snapshot = SnapshotData::new(vec![0u8; 200]); // Too large

        manager.cache_snapshot(key.clone(), large_snapshot);
        assert!(!manager.is_cached(&key));
    }

    #[test]
    fn test_metrics_recording() {
        let mut manager = FastResetManager::default_config();

        manager.record_full_reset(Duration::from_millis(100));
        manager.record_full_reset(Duration::from_millis(100));
        manager.record_fast_reset(Duration::from_millis(10));

        assert_eq!(manager.metrics.full_resets, 2);
        assert_eq!(manager.metrics.fast_resets, 1);
    }

    #[test]
    fn test_speedup_ratio() {
        let mut manager = FastResetManager::default_config();

        // Full resets: 100ms average
        for _ in 0..10 {
            manager.record_full_reset(Duration::from_millis(100));
        }

        // Fast resets: 10ms average
        for _ in 0..10 {
            manager.record_fast_reset(Duration::from_millis(10));
        }

        let speedup = manager.metrics.speedup_ratio();
        assert!(speedup > 9.0 && speedup < 11.0, "speedup = {}", speedup);
    }

    #[test]
    fn test_fast_reset_rate() {
        let mut manager = FastResetManager::default_config();

        manager.record_full_reset(Duration::from_millis(100));
        manager.record_fast_reset(Duration::from_millis(10));
        manager.record_fast_reset(Duration::from_millis(10));
        manager.record_fast_reset(Duration::from_millis(10));

        let rate = manager.metrics.fast_reset_rate();
        assert!((rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_reset_result() {
        let obs = Tensor::zeros(DType::Float32, vec![4]);
        let result = ResetResult::new(obs, true, Duration::from_millis(5));

        assert!(result.was_fast);
        assert_eq!(result.duration, Duration::from_millis(5));
    }
}
