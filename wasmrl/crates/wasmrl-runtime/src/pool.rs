// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Instance pool management for WasmRL runtime.
//!
//! The instance pool manages a collection of environment instances,
//! providing allocation, tracking, and recycling functionality.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::error::{RuntimeError, RuntimeResult};
use crate::instance::{InstanceHandle, InstanceInfo, InstanceStatus};

/// Statistics about pool state.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of instances created.
    pub total_created: u64,
    /// Number of currently active instances.
    pub active: usize,
    /// Number of instances in ready queue.
    pub ready: usize,
    /// Number of instances in error state.
    pub errored: usize,
    /// Number of instances recycled.
    pub recycled: u64,
    /// Maximum capacity.
    pub capacity: usize,
}

/// Instance pool for managing environment instances.
#[derive(Debug)]
pub struct InstancePool {
    /// Maximum number of instances.
    capacity: usize,
    /// Active instances (currently in use or ready).
    instances: HashMap<u64, InstanceInfo>,
    /// Queue of ready instances (LIFO for cache locality).
    ready_queue: VecDeque<InstanceHandle>,
    /// Counter for total instances created.
    total_created: u64,
    /// Counter for recycled instances.
    recycled: u64,
}

impl InstancePool {
    /// Create a new instance pool with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            instances: HashMap::with_capacity(capacity),
            ready_queue: VecDeque::with_capacity(capacity),
            total_created: 0,
            recycled: 0,
        }
    }

    /// Get current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let errored = self
            .instances
            .values()
            .filter(|i| i.status.is_error())
            .count();

        PoolStats {
            total_created: self.total_created,
            active: self.instances.len(),
            ready: self.ready_queue.len(),
            errored,
            recycled: self.recycled,
            capacity: self.capacity,
        }
    }

    /// Allocate a new instance handle.
    ///
    /// Returns an existing ready instance if available, otherwise creates new.
    /// Returns error if pool is at capacity with no ready instances.
    pub fn allocate(&mut self) -> RuntimeResult<InstanceHandle> {
        // Try to get a ready instance first
        if let Some(handle) = self.ready_queue.pop_back() {
            if let Some(info) = self.instances.get_mut(&handle.id) {
                info.status = InstanceStatus::Running;
                return Ok(handle);
            }
        }

        // Create new instance if under capacity
        if self.instances.len() < self.capacity {
            let handle = InstanceHandle::new();
            let mut info = InstanceInfo::new(handle);
            info.status = InstanceStatus::Running;
            self.instances.insert(handle.id, info);
            self.total_created += 1;
            return Ok(handle);
        }

        // Pool exhausted
        Err(RuntimeError::pool_exhausted(self.capacity))
    }

    /// Allocate multiple instances at once.
    pub fn allocate_many(&mut self, count: usize) -> RuntimeResult<Vec<InstanceHandle>> {
        // Check if we have enough capacity
        let available = self.capacity - self.instances.len() + self.ready_queue.len();
        if count > available {
            return Err(RuntimeError::pool_exhausted(self.capacity));
        }

        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            handles.push(self.allocate()?);
        }
        Ok(handles)
    }

    /// Release an instance back to the pool.
    ///
    /// The instance is marked as ready and added to the ready queue.
    pub fn release(&mut self, handle: InstanceHandle) -> RuntimeResult<()> {
        let info = self
            .instances
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        if info.status == InstanceStatus::Running {
            info.status = InstanceStatus::Ready;
            self.ready_queue.push_back(handle);
        }

        Ok(())
    }

    /// Mark an instance as having an error.
    pub fn mark_error(&mut self, handle: InstanceHandle, fatal: bool) -> RuntimeResult<()> {
        let info = self
            .instances
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        info.status = if fatal {
            InstanceStatus::ErrorFatal
        } else {
            InstanceStatus::ErrorRecoverable
        };

        Ok(())
    }

    /// Recycle an errored or terminated instance.
    ///
    /// Removes the instance from the pool, allowing a new one to be created.
    pub fn recycle(&mut self, handle: InstanceHandle) -> RuntimeResult<InstanceInfo> {
        let info = self
            .instances
            .remove(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;

        if !info.status.can_recycle() {
            // Put it back if not recyclable
            self.instances.insert(handle.id, info.clone());
            return Err(RuntimeError::execution(format!(
                "Cannot recycle instance in state: {}",
                info.status
            )));
        }

        // Remove from ready queue if present
        self.ready_queue.retain(|h| h.id != handle.id);

        self.recycled += 1;
        Ok(info)
    }

    /// Get information about an instance.
    pub fn get_info(&self, handle: InstanceHandle) -> Option<&InstanceInfo> {
        self.instances.get(&handle.id)
    }

    /// Get mutable information about an instance.
    pub fn get_info_mut(&mut self, handle: InstanceHandle) -> Option<&mut InstanceInfo> {
        self.instances.get_mut(&handle.id)
    }

    /// Update instance step count.
    pub fn record_step(&mut self, handle: InstanceHandle) -> RuntimeResult<()> {
        let info = self
            .instances
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;
        info.step_count += 1;
        Ok(())
    }

    /// Update instance reset count.
    pub fn record_reset(&mut self, handle: InstanceHandle) -> RuntimeResult<()> {
        let info = self
            .instances
            .get_mut(&handle.id)
            .ok_or_else(|| RuntimeError::instance_not_found(handle.id))?;
        info.reset_count += 1;
        info.episode += 1;
        Ok(())
    }

    /// Get all active instance handles.
    pub fn active_handles(&self) -> Vec<InstanceHandle> {
        self.instances
            .keys()
            .map(|&id| InstanceHandle { id })
            .collect()
    }

    /// Get number of available (ready or allocatable) slots.
    pub fn available(&self) -> usize {
        let new_slots = self.capacity.saturating_sub(self.instances.len());
        new_slots + self.ready_queue.len()
    }

    /// Check if pool is at capacity.
    pub fn is_full(&self) -> bool {
        self.instances.len() >= self.capacity && self.ready_queue.is_empty()
    }

    /// Clear all instances from the pool.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.ready_queue.clear();
    }
}

/// Thread-safe wrapper around InstancePool.
#[derive(Debug, Clone)]
pub struct SharedPool {
    inner: Arc<Mutex<InstancePool>>,
}

impl SharedPool {
    /// Create a new shared pool.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(InstancePool::new(capacity))),
        }
    }

    /// Allocate an instance.
    pub fn allocate(&self) -> RuntimeResult<InstanceHandle> {
        self.inner.lock().unwrap().allocate()
    }

    /// Allocate multiple instances.
    pub fn allocate_many(&self, count: usize) -> RuntimeResult<Vec<InstanceHandle>> {
        self.inner.lock().unwrap().allocate_many(count)
    }

    /// Release an instance.
    pub fn release(&self, handle: InstanceHandle) -> RuntimeResult<()> {
        self.inner.lock().unwrap().release(handle)
    }

    /// Mark an instance as having an error.
    pub fn mark_error(&self, handle: InstanceHandle, fatal: bool) -> RuntimeResult<()> {
        self.inner.lock().unwrap().mark_error(handle, fatal)
    }

    /// Recycle an instance.
    pub fn recycle(&self, handle: InstanceHandle) -> RuntimeResult<InstanceInfo> {
        self.inner.lock().unwrap().recycle(handle)
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.inner.lock().unwrap().stats()
    }

    /// Get instance info.
    pub fn get_info(&self, handle: InstanceHandle) -> Option<InstanceInfo> {
        self.inner.lock().unwrap().get_info(handle).cloned()
    }

    /// Record a step.
    pub fn record_step(&self, handle: InstanceHandle) -> RuntimeResult<()> {
        self.inner.lock().unwrap().record_step(handle)
    }

    /// Record a reset.
    pub fn record_reset(&self, handle: InstanceHandle) -> RuntimeResult<()> {
        self.inner.lock().unwrap().record_reset(handle)
    }

    /// Check if pool is full.
    pub fn is_full(&self) -> bool {
        self.inner.lock().unwrap().is_full()
    }

    /// Get available count.
    pub fn available(&self) -> usize {
        self.inner.lock().unwrap().available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocate() {
        let mut pool = InstancePool::new(10);
        let handle = pool.allocate().unwrap();
        assert!(handle.id > 0);

        let stats = pool.stats();
        assert_eq!(stats.active, 1);
        assert_eq!(stats.total_created, 1);
    }

    #[test]
    fn test_pool_allocate_many() {
        let mut pool = InstancePool::new(10);
        let handles = pool.allocate_many(5).unwrap();
        assert_eq!(handles.len(), 5);

        let stats = pool.stats();
        assert_eq!(stats.active, 5);
    }

    #[test]
    fn test_pool_capacity_limit() {
        let mut pool = InstancePool::new(2);
        let _h1 = pool.allocate().unwrap();
        let _h2 = pool.allocate().unwrap();

        let result = pool.allocate();
        assert!(result.is_err());
        if let Err(RuntimeError::PoolExhausted(cap)) = result {
            assert_eq!(cap, 2);
        } else {
            panic!("Expected PoolExhausted error");
        }
    }

    #[test]
    fn test_pool_release_and_reuse() {
        let mut pool = InstancePool::new(2);
        let h1 = pool.allocate().unwrap();
        let _h2 = pool.allocate().unwrap();

        // Release h1
        pool.release(h1).unwrap();

        // Should be able to allocate again (reuses h1)
        let h3 = pool.allocate().unwrap();
        assert_eq!(h3.id, h1.id);
    }

    #[test]
    fn test_pool_mark_error() {
        let mut pool = InstancePool::new(10);
        let handle = pool.allocate().unwrap();

        pool.mark_error(handle, true).unwrap();

        let info = pool.get_info(handle).unwrap();
        assert_eq!(info.status, InstanceStatus::ErrorFatal);
    }

    #[test]
    fn test_pool_recycle() {
        let mut pool = InstancePool::new(2);
        let h1 = pool.allocate().unwrap();
        let _h2 = pool.allocate().unwrap();

        // Mark as fatal error
        pool.mark_error(h1, true).unwrap();

        // Recycle it
        let recycled = pool.recycle(h1).unwrap();
        assert_eq!(recycled.handle, h1);

        // Now we should be able to allocate again
        let h3 = pool.allocate().unwrap();
        assert_ne!(h3.id, h1.id); // New instance created
    }

    #[test]
    fn test_pool_record_step() {
        let mut pool = InstancePool::new(10);
        let handle = pool.allocate().unwrap();

        for _ in 0..5 {
            pool.record_step(handle).unwrap();
        }

        let info = pool.get_info(handle).unwrap();
        assert_eq!(info.step_count, 5);
    }

    #[test]
    fn test_pool_record_reset() {
        let mut pool = InstancePool::new(10);
        let handle = pool.allocate().unwrap();

        pool.record_reset(handle).unwrap();
        pool.record_reset(handle).unwrap();

        let info = pool.get_info(handle).unwrap();
        assert_eq!(info.reset_count, 2);
        assert_eq!(info.episode, 2);
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = InstancePool::new(10);
        let h1 = pool.allocate().unwrap();
        let h2 = pool.allocate().unwrap();
        pool.release(h1).unwrap();
        pool.mark_error(h2, true).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.active, 2);
        assert_eq!(stats.ready, 1);
        assert_eq!(stats.errored, 1);
        assert_eq!(stats.capacity, 10);
    }

    #[test]
    fn test_pool_available() {
        let mut pool = InstancePool::new(10);
        assert_eq!(pool.available(), 10);

        let h1 = pool.allocate().unwrap();
        assert_eq!(pool.available(), 9);

        pool.release(h1).unwrap();
        assert_eq!(pool.available(), 10);
    }

    #[test]
    fn test_pool_is_full() {
        let mut pool = InstancePool::new(2);
        assert!(!pool.is_full());

        let _h1 = pool.allocate().unwrap();
        assert!(!pool.is_full());

        let _h2 = pool.allocate().unwrap();
        assert!(pool.is_full());
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = InstancePool::new(10);
        let _h1 = pool.allocate().unwrap();
        let h2 = pool.allocate().unwrap();
        pool.release(h2).unwrap();

        pool.clear();

        let stats = pool.stats();
        assert_eq!(stats.active, 0);
        assert_eq!(stats.ready, 0);
    }

    #[test]
    fn test_shared_pool() {
        let pool = SharedPool::new(10);

        let h1 = pool.allocate().unwrap();
        let h2 = pool.allocate().unwrap();

        pool.release(h1).unwrap();
        pool.record_step(h2).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.active, 2);
        assert_eq!(stats.ready, 1);
    }

    #[test]
    fn test_shared_pool_thread_safe() {
        use std::thread;

        let pool = SharedPool::new(100);
        let mut handles = vec![];

        for _ in 0..4 {
            let pool_clone = pool.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let h = pool_clone.allocate().unwrap();
                    pool_clone.record_step(h).unwrap();
                    pool_clone.release(h).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let stats = pool.stats();
        // All should be released back
        assert_eq!(stats.active, stats.ready);
    }
}
