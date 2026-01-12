// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Snapshot cache for fast environment reset.
//!
//! This module provides caching of initial snapshots to enable fast reset
//! operations. Instead of re-initializing environments from scratch, we
//! restore from a cached snapshot of the initial state.
//!
//! # Usage
//!
//! ```ignore
//! use wasmrl_runtime::{SnapshotCache, SnapshotKey};
//!
//! let mut cache = SnapshotCache::new(1000); // 1000 entries max
//!
//! // Cache an initial snapshot
//! let key = SnapshotKey::new("counter_env", "{}", 42);
//! cache.put(key.clone(), snapshot);
//!
//! // Fast reset using cached snapshot
//! if let Some(snapshot) = cache.get(&key) {
//!     runtime.restore(handle, snapshot)?;
//! }
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use wasmrl_wit::SnapshotData;

/// Key for identifying cached snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnapshotKey {
    /// Component identifier (file path or OCI reference).
    pub component_id: String,
    /// Configuration hash or JSON string.
    pub config_hash: String,
    /// Initial seed used for reset.
    pub seed: u64,
}

impl SnapshotKey {
    /// Create a new snapshot key.
    pub fn new(component_id: impl Into<String>, config: impl Into<String>, seed: u64) -> Self {
        Self {
            component_id: component_id.into(),
            config_hash: config.into(),
            seed,
        }
    }

    /// Create a key from component ID and seed only (empty config).
    pub fn from_id_seed(component_id: impl Into<String>, seed: u64) -> Self {
        Self::new(component_id, "", seed)
    }
}

/// Cached snapshot entry with metadata.
#[derive(Debug, Clone)]
pub struct CachedSnapshot {
    /// The snapshot data.
    pub data: SnapshotData,
    /// When this snapshot was created.
    pub created_at: Instant,
    /// Number of times this snapshot has been used.
    pub use_count: u64,
    /// Last time this snapshot was accessed.
    pub last_accessed: Instant,
    /// Size of the snapshot data in bytes.
    pub size_bytes: usize,
}

impl CachedSnapshot {
    /// Create a new cached snapshot.
    pub fn new(data: SnapshotData) -> Self {
        let size_bytes = data.data.len();
        let now = Instant::now();
        Self {
            data,
            created_at: now,
            use_count: 0,
            last_accessed: now,
            size_bytes,
        }
    }

    /// Mark this snapshot as accessed.
    pub fn touch(&mut self) {
        self.use_count += 1;
        self.last_accessed = Instant::now();
    }

    /// Get age of this snapshot.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last access.
    pub fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }
}

/// Statistics about the snapshot cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of entries in the cache.
    pub entries: usize,
    /// Total size of cached data in bytes.
    pub total_bytes: usize,
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Maximum capacity.
    pub capacity: usize,
}

impl CacheStats {
    /// Calculate hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// LRU-based snapshot cache.
#[derive(Debug)]
pub struct SnapshotCache {
    /// Cached snapshots by key.
    entries: HashMap<SnapshotKey, CachedSnapshot>,
    /// Maximum number of entries.
    capacity: usize,
    /// Maximum total size in bytes (0 = unlimited).
    max_bytes: usize,
    /// Total size of cached data.
    total_bytes: usize,
    /// Cache statistics.
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl SnapshotCache {
    /// Create a new snapshot cache with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            max_bytes: 0,
            total_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Create a cache with both entry and byte limits.
    pub fn with_byte_limit(capacity: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            max_bytes,
            total_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get a snapshot from the cache.
    pub fn get(&mut self, key: &SnapshotKey) -> Option<&SnapshotData> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch();
            self.hits += 1;
            Some(&entry.data)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Get a snapshot without updating access time.
    pub fn peek(&self, key: &SnapshotKey) -> Option<&SnapshotData> {
        self.entries.get(key).map(|e| &e.data)
    }

    /// Put a snapshot into the cache.
    pub fn put(&mut self, key: SnapshotKey, data: SnapshotData) {
        let entry = CachedSnapshot::new(data);
        let size = entry.size_bytes;

        // Evict if necessary
        while self.should_evict(size) {
            if !self.evict_one() {
                break;
            }
        }

        // Update totals
        if let Some(old) = self.entries.insert(key, entry) {
            self.total_bytes -= old.size_bytes;
        }
        self.total_bytes += size;
    }

    /// Check if a key exists in the cache.
    pub fn contains(&self, key: &SnapshotKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove a snapshot from the cache.
    pub fn remove(&mut self, key: &SnapshotKey) -> Option<SnapshotData> {
        if let Some(entry) = self.entries.remove(key) {
            self.total_bytes -= entry.size_bytes;
            Some(entry.data)
        } else {
            None
        }
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            total_bytes: self.total_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            capacity: self.capacity,
        }
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if we need to evict before adding an entry of given size.
    fn should_evict(&self, new_size: usize) -> bool {
        if self.entries.len() >= self.capacity {
            return true;
        }
        if self.max_bytes > 0 && self.total_bytes + new_size > self.max_bytes {
            return true;
        }
        false
    }

    /// Evict the least recently used entry.
    fn evict_one(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        // Find LRU entry
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes -= entry.size_bytes;
                self.evictions += 1;
                return true;
            }
        }

        false
    }

    /// Evict entries older than given duration.
    pub fn evict_older_than(&mut self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut evicted = 0;

        let keys_to_remove: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, v)| now.duration_since(v.created_at) > max_age)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes -= entry.size_bytes;
                self.evictions += 1;
                evicted += 1;
            }
        }

        evicted
    }

    /// Evict entries not accessed within given duration.
    pub fn evict_idle(&mut self, max_idle: Duration) -> usize {
        let now = Instant::now();
        let mut evicted = 0;

        let keys_to_remove: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, v)| now.duration_since(v.last_accessed) > max_idle)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes -= entry.size_bytes;
                self.evictions += 1;
                evicted += 1;
            }
        }

        evicted
    }
}

/// Thread-safe wrapper around SnapshotCache.
#[derive(Debug, Clone)]
pub struct SharedSnapshotCache {
    inner: Arc<RwLock<SnapshotCache>>,
}

impl SharedSnapshotCache {
    /// Create a new shared cache.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SnapshotCache::new(capacity))),
        }
    }

    /// Create with byte limit.
    pub fn with_byte_limit(capacity: usize, max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SnapshotCache::with_byte_limit(
                capacity, max_bytes,
            ))),
        }
    }

    /// Get a snapshot (cloned).
    pub fn get(&self, key: &SnapshotKey) -> Option<SnapshotData> {
        self.inner.write().ok()?.get(key).cloned()
    }

    /// Put a snapshot.
    pub fn put(&self, key: SnapshotKey, data: SnapshotData) {
        if let Ok(mut cache) = self.inner.write() {
            cache.put(key, data);
        }
    }

    /// Check if key exists.
    pub fn contains(&self, key: &SnapshotKey) -> bool {
        self.inner
            .read()
            .map(|c| c.contains(key))
            .unwrap_or(false)
    }

    /// Remove a snapshot.
    pub fn remove(&self, key: &SnapshotKey) -> Option<SnapshotData> {
        self.inner.write().ok()?.remove(key)
    }

    /// Get stats.
    pub fn stats(&self) -> CacheStats {
        self.inner
            .read()
            .map(|c| c.stats())
            .unwrap_or_default()
    }

    /// Clear the cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.inner.write() {
            cache.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(size: usize) -> SnapshotData {
        SnapshotData::new(vec![0u8; size])
    }

    #[test]
    fn test_snapshot_key_creation() {
        let key = SnapshotKey::new("counter_env", "{\"target\": 10}", 42);
        assert_eq!(key.component_id, "counter_env");
        assert_eq!(key.seed, 42);

        let key2 = SnapshotKey::from_id_seed("env", 123);
        assert_eq!(key2.config_hash, "");
        assert_eq!(key2.seed, 123);
    }

    #[test]
    fn test_snapshot_key_equality() {
        let k1 = SnapshotKey::new("env", "cfg", 1);
        let k2 = SnapshotKey::new("env", "cfg", 1);
        let k3 = SnapshotKey::new("env", "cfg", 2);

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_cached_snapshot_metadata() {
        let mut entry = CachedSnapshot::new(make_snapshot(100));
        assert_eq!(entry.size_bytes, 100);
        assert_eq!(entry.use_count, 0);

        entry.touch();
        assert_eq!(entry.use_count, 1);

        entry.touch();
        assert_eq!(entry.use_count, 2);
    }

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = SnapshotCache::new(10);

        let key = SnapshotKey::new("env", "cfg", 42);
        let snapshot = make_snapshot(100);

        // Put and get
        cache.put(key.clone(), snapshot.clone());
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.data.len(), 100);

        // Stats
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = SnapshotCache::new(10);

        let key = SnapshotKey::new("env", "cfg", 42);
        assert!(cache.get(&key).is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_cache_eviction_by_capacity() {
        let mut cache = SnapshotCache::new(3);

        // Fill cache
        for i in 0..3 {
            let key = SnapshotKey::new("env", "cfg", i);
            cache.put(key, make_snapshot(10));
        }
        assert_eq!(cache.len(), 3);

        // Add one more - should evict
        let key4 = SnapshotKey::new("env", "cfg", 100);
        cache.put(key4, make_snapshot(10));
        assert_eq!(cache.len(), 3);

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = SnapshotCache::new(3);

        // Add entries
        let k1 = SnapshotKey::new("env", "cfg", 1);
        let k2 = SnapshotKey::new("env", "cfg", 2);
        let k3 = SnapshotKey::new("env", "cfg", 3);

        cache.put(k1.clone(), make_snapshot(10));
        cache.put(k2.clone(), make_snapshot(10));
        cache.put(k3.clone(), make_snapshot(10));

        // Access k1 and k3 (k2 becomes LRU)
        cache.get(&k1);
        cache.get(&k3);

        // Add new entry - should evict k2
        let k4 = SnapshotKey::new("env", "cfg", 4);
        cache.put(k4.clone(), make_snapshot(10));

        assert!(cache.contains(&k1));
        assert!(!cache.contains(&k2)); // Evicted
        assert!(cache.contains(&k3));
        assert!(cache.contains(&k4));
    }

    #[test]
    fn test_cache_byte_limit() {
        let mut cache = SnapshotCache::with_byte_limit(100, 250);

        // Add entries (each 100 bytes)
        for i in 0..3 {
            let key = SnapshotKey::new("env", "cfg", i);
            cache.put(key, make_snapshot(100));
        }

        // Should have evicted to stay under 250 bytes
        assert!(cache.total_bytes <= 250);
        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = SnapshotCache::new(10);

        let key = SnapshotKey::new("env", "cfg", 42);
        cache.put(key.clone(), make_snapshot(100));
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(&key);
        assert!(removed.is_some());
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(&key));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = SnapshotCache::new(10);

        for i in 0..5 {
            cache.put(SnapshotKey::new("env", "cfg", i), make_snapshot(100));
        }
        assert_eq!(cache.len(), 5);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = SnapshotCache::new(10);

        let key = SnapshotKey::new("env", "cfg", 42);
        cache.put(key.clone(), make_snapshot(100));

        // 3 hits
        cache.get(&key);
        cache.get(&key);
        cache.get(&key);

        // 1 miss
        cache.get(&SnapshotKey::new("env", "cfg", 999));

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_shared_cache_thread_safe() {
        use std::thread;

        let cache = SharedSnapshotCache::new(100);
        let mut handles = vec![];

        // Multiple threads writing and reading
        for i in 0..4 {
            let cache_clone = cache.clone();
            handles.push(thread::spawn(move || {
                for j in 0..25 {
                    let key = SnapshotKey::new("env", "cfg", (i * 25 + j) as u64);
                    cache_clone.put(key.clone(), make_snapshot(10));
                    cache_clone.get(&key);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let stats = cache.stats();
        assert!(stats.entries <= 100);
    }

    #[test]
    fn test_peek_does_not_update_stats() {
        let mut cache = SnapshotCache::new(10);

        let key = SnapshotKey::new("env", "cfg", 42);
        cache.put(key.clone(), make_snapshot(100));

        // Peek should not update hits
        let _ = cache.peek(&key);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);

        // Get should update hits
        let _ = cache.get(&key);
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
    }
}
