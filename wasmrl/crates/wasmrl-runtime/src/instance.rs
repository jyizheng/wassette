// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Environment instance management for WasmRL runtime.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global instance ID counter.
static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique instance ID.
fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::SeqCst)
}

/// Handle to a WasmRL environment instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceHandle {
    /// Unique instance ID.
    pub id: u64,
}

impl InstanceHandle {
    /// Create a new instance handle with a unique ID.
    pub fn new() -> Self {
        Self {
            id: next_instance_id(),
        }
    }

    /// Create an instance handle with a specific ID (for testing).
    #[cfg(test)]
    pub fn with_id(id: u64) -> Self {
        Self { id }
    }
}

impl Default for InstanceHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InstanceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instance({})", self.id)
    }
}

/// Status of a WasmRL instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceStatus {
    /// Instance is uninitialized (just created).
    Uninitialized,
    /// Instance is ready for use.
    Ready,
    /// Instance is currently executing.
    Running,
    /// Instance is paused (snapshot available).
    Paused,
    /// Instance has encountered a recoverable error.
    ErrorRecoverable,
    /// Instance has encountered a fatal error and needs recycling.
    ErrorFatal,
    /// Instance has been terminated.
    Terminated,
}

impl InstanceStatus {
    /// Check if instance can accept new operations.
    pub fn is_available(&self) -> bool {
        matches!(self, InstanceStatus::Ready | InstanceStatus::Paused)
    }

    /// Check if instance is in an error state.
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            InstanceStatus::ErrorRecoverable | InstanceStatus::ErrorFatal
        )
    }

    /// Check if instance can be recycled.
    pub fn can_recycle(&self) -> bool {
        matches!(
            self,
            InstanceStatus::ErrorRecoverable
                | InstanceStatus::ErrorFatal
                | InstanceStatus::Terminated
        )
    }
}

impl fmt::Display for InstanceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstanceStatus::Uninitialized => write!(f, "uninitialized"),
            InstanceStatus::Ready => write!(f, "ready"),
            InstanceStatus::Running => write!(f, "running"),
            InstanceStatus::Paused => write!(f, "paused"),
            InstanceStatus::ErrorRecoverable => write!(f, "error (recoverable)"),
            InstanceStatus::ErrorFatal => write!(f, "error (fatal)"),
            InstanceStatus::Terminated => write!(f, "terminated"),
        }
    }
}

/// Information about an instance.
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// Instance handle.
    pub handle: InstanceHandle,
    /// Current status.
    pub status: InstanceStatus,
    /// Number of steps executed.
    pub step_count: u64,
    /// Number of resets performed.
    pub reset_count: u64,
    /// Current episode number.
    pub episode: u64,
    /// Memory usage in bytes (approximate).
    pub memory_bytes: u64,
    /// Fuel consumed in current operation.
    pub fuel_consumed: u64,
}

impl InstanceInfo {
    /// Create info for a new instance.
    pub fn new(handle: InstanceHandle) -> Self {
        Self {
            handle,
            status: InstanceStatus::Uninitialized,
            step_count: 0,
            reset_count: 0,
            episode: 0,
            memory_bytes: 0,
            fuel_consumed: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_handle_unique() {
        let h1 = InstanceHandle::new();
        let h2 = InstanceHandle::new();
        assert_ne!(h1.id, h2.id);
    }

    #[test]
    fn test_instance_handle_display() {
        let handle = InstanceHandle::with_id(42);
        assert_eq!(format!("{}", handle), "Instance(42)");
    }

    #[test]
    fn test_instance_handle_equality() {
        let h1 = InstanceHandle::with_id(1);
        let h2 = InstanceHandle::with_id(1);
        let h3 = InstanceHandle::with_id(2);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_instance_status_available() {
        assert!(InstanceStatus::Ready.is_available());
        assert!(InstanceStatus::Paused.is_available());
        assert!(!InstanceStatus::Running.is_available());
        assert!(!InstanceStatus::ErrorFatal.is_available());
    }

    #[test]
    fn test_instance_status_error() {
        assert!(InstanceStatus::ErrorRecoverable.is_error());
        assert!(InstanceStatus::ErrorFatal.is_error());
        assert!(!InstanceStatus::Ready.is_error());
    }

    #[test]
    fn test_instance_status_can_recycle() {
        assert!(InstanceStatus::ErrorFatal.can_recycle());
        assert!(InstanceStatus::Terminated.can_recycle());
        assert!(!InstanceStatus::Ready.can_recycle());
        assert!(!InstanceStatus::Running.can_recycle());
    }

    #[test]
    fn test_instance_status_display() {
        assert_eq!(format!("{}", InstanceStatus::Ready), "ready");
        assert_eq!(format!("{}", InstanceStatus::ErrorFatal), "error (fatal)");
    }

    #[test]
    fn test_instance_info_new() {
        let handle = InstanceHandle::new();
        let info = InstanceInfo::new(handle);
        assert_eq!(info.handle, handle);
        assert_eq!(info.status, InstanceStatus::Uninitialized);
        assert_eq!(info.step_count, 0);
        assert_eq!(info.episode, 0);
    }
}
