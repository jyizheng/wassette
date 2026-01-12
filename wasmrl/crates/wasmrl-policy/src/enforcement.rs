// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Budget enforcement for WasmRL environments.

use crate::{Policy, PolicyError, PolicyResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of an enforcement check.
#[derive(Debug, Clone)]
pub enum EnforcementResult {
    /// Operation is allowed to proceed.
    Allowed,
    /// Operation was denied due to budget/capability.
    Denied(EnforcementAction),
    /// Operation exceeded budget but was completed.
    Exceeded(EnforcementAction),
}

impl EnforcementResult {
    /// Check if the result allows continuation.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// Check if the result is a denial.
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied(_))
    }

    /// Get the action if denied or exceeded.
    pub fn action(&self) -> Option<&EnforcementAction> {
        match self {
            Self::Denied(a) | Self::Exceeded(a) => Some(a),
            Self::Allowed => None,
        }
    }
}

/// Action taken by enforcement.
#[derive(Debug, Clone)]
pub enum EnforcementAction {
    /// Fuel budget exhausted.
    FuelExhausted {
        /// Operation that exhausted fuel.
        operation: String,
        /// Fuel consumed.
        consumed: u64,
        /// Budget limit.
        budget: u64,
    },
    /// Timeout exceeded.
    Timeout {
        /// Operation that timed out.
        operation: String,
        /// Elapsed time.
        elapsed: Duration,
        /// Timeout limit.
        limit: Duration,
    },
    /// Memory limit exceeded.
    MemoryExceeded {
        /// Current memory usage in bytes.
        current: u64,
        /// Memory limit in bytes.
        limit: u64,
    },
    /// Capability denied.
    CapabilityDenied {
        /// Capability that was denied.
        capability: String,
        /// Reason for denial.
        reason: String,
    },
    /// Trapped/crashed.
    Trap {
        /// Trap message.
        message: String,
    },
}

impl EnforcementAction {
    /// Convert to PolicyError.
    pub fn to_error(&self) -> PolicyError {
        match self {
            Self::FuelExhausted {
                operation: _,
                consumed,
                budget,
            } => PolicyError::budget_exceeded("fuel", *budget, *consumed),
            Self::Timeout {
                operation,
                elapsed,
                limit,
            } => {
                PolicyError::timeout_exceeded(operation, elapsed.as_millis() as u64, limit.as_millis() as u64)
            }
            Self::MemoryExceeded { current, limit } => {
                PolicyError::memory_exceeded((*current / (1024 * 1024)) as u32, (*limit / (1024 * 1024)) as u32)
            }
            Self::CapabilityDenied { capability, .. } => PolicyError::capability_denied(capability),
            Self::Trap { message } => PolicyError::EnforcementError(format!("Trap: {}", message)),
        }
    }
}

/// Enforcer for policy budgets and capabilities.
#[derive(Debug)]
pub struct BudgetEnforcer {
    /// Policy being enforced.
    policy: Policy,
    /// Total fuel consumed.
    fuel_consumed: AtomicU64,
    /// Number of steps executed.
    steps_executed: AtomicU64,
    /// Number of resets executed.
    resets_executed: AtomicU64,
    /// Start time for timeout tracking.
    start_time: Instant,
    /// Operation deadline.
    deadline: Option<Instant>,
}

impl BudgetEnforcer {
    /// Create a new enforcer for the given policy.
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            fuel_consumed: AtomicU64::new(0),
            steps_executed: AtomicU64::new(0),
            resets_executed: AtomicU64::new(0),
            start_time: Instant::now(),
            deadline: None,
        }
    }

    /// Create an enforcer wrapped in Arc for sharing.
    pub fn shared(policy: Policy) -> Arc<Self> {
        Arc::new(Self::new(policy))
    }

    /// Get the policy being enforced.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Get total fuel consumed.
    pub fn fuel_consumed(&self) -> u64 {
        self.fuel_consumed.load(Ordering::Relaxed)
    }

    /// Get number of steps executed.
    pub fn steps_executed(&self) -> u64 {
        self.steps_executed.load(Ordering::Relaxed)
    }

    /// Get number of resets executed.
    pub fn resets_executed(&self) -> u64 {
        self.resets_executed.load(Ordering::Relaxed)
    }

    /// Set operation deadline.
    pub fn set_deadline(&mut self, timeout: Duration) {
        self.deadline = Some(Instant::now() + timeout);
    }

    /// Clear operation deadline.
    pub fn clear_deadline(&mut self) {
        self.deadline = None;
    }

    /// Check if deadline has passed.
    pub fn is_past_deadline(&self) -> bool {
        self.deadline.map(|d| Instant::now() > d).unwrap_or(false)
    }

    /// Get remaining time until deadline.
    pub fn remaining_time(&self) -> Option<Duration> {
        self.deadline.map(|d| d.saturating_duration_since(Instant::now()))
    }

    // =========================================================================
    // Pre-operation Checks
    // =========================================================================

    /// Check if a step operation is allowed.
    pub fn check_step(&self) -> EnforcementResult {
        // Check timeout
        if self.policy.timeout.enabled {
            if let Some(deadline) = self.deadline {
                if Instant::now() > deadline {
                    return EnforcementResult::Denied(EnforcementAction::Timeout {
                        operation: "step".to_string(),
                        elapsed: self.start_time.elapsed(),
                        limit: self.policy.timeout.step_duration(),
                    });
                }
            }
        }

        EnforcementResult::Allowed
    }

    /// Check if a reset operation is allowed.
    pub fn check_reset(&self) -> EnforcementResult {
        // Check timeout
        if self.policy.timeout.enabled {
            if let Some(deadline) = self.deadline {
                if Instant::now() > deadline {
                    return EnforcementResult::Denied(EnforcementAction::Timeout {
                        operation: "reset".to_string(),
                        elapsed: self.start_time.elapsed(),
                        limit: self.policy.timeout.reset_duration(),
                    });
                }
            }
        }

        EnforcementResult::Allowed
    }

    /// Check if memory usage is within limits.
    pub fn check_memory(&self, current_bytes: u64) -> EnforcementResult {
        if !self.policy.memory.enabled {
            return EnforcementResult::Allowed;
        }

        let limit = self.policy.memory.max_bytes();
        if current_bytes > limit {
            return EnforcementResult::Denied(EnforcementAction::MemoryExceeded {
                current: current_bytes,
                limit,
            });
        }

        EnforcementResult::Allowed
    }

    /// Check if a filesystem read is allowed.
    pub fn check_read(&self, path: &str) -> EnforcementResult {
        if self.policy.allows_read(path) {
            EnforcementResult::Allowed
        } else {
            EnforcementResult::Denied(EnforcementAction::CapabilityDenied {
                capability: "filesystem_read".to_string(),
                reason: format!("Path not in allowlist: {}", path),
            })
        }
    }

    /// Check if a filesystem write is allowed.
    pub fn check_write(&self, path: &str) -> EnforcementResult {
        if self.policy.allows_write(path) {
            EnforcementResult::Allowed
        } else {
            EnforcementResult::Denied(EnforcementAction::CapabilityDenied {
                capability: "filesystem_write".to_string(),
                reason: format!("Path not in allowlist: {}", path),
            })
        }
    }

    /// Check if network access is allowed.
    pub fn check_network(&self) -> EnforcementResult {
        if self.policy.allows_network() {
            EnforcementResult::Allowed
        } else {
            EnforcementResult::Denied(EnforcementAction::CapabilityDenied {
                capability: "network".to_string(),
                reason: "Network access is disabled by policy".to_string(),
            })
        }
    }

    // =========================================================================
    // Post-operation Recording
    // =========================================================================

    /// Record fuel consumed by an operation.
    pub fn record_fuel(&self, consumed: u64) {
        self.fuel_consumed.fetch_add(consumed, Ordering::Relaxed);
    }

    /// Record a step operation.
    pub fn record_step(&self) {
        self.steps_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a reset operation.
    pub fn record_reset(&self) {
        self.resets_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Check fuel consumption against budget.
    pub fn check_fuel(&self, operation: &str) -> EnforcementResult {
        if !self.policy.fuel.enabled {
            return EnforcementResult::Allowed;
        }

        let consumed = self.fuel_consumed.load(Ordering::Relaxed);
        let budget = match operation {
            "step" => self.policy.fuel.per_step,
            "reset" => self.policy.fuel.per_reset,
            "batch" => self.policy.fuel.per_batch,
            "init" => self.policy.fuel.per_init,
            "snapshot" => self.policy.fuel.per_snapshot,
            "restore" => self.policy.fuel.per_restore,
            _ => self.policy.fuel.per_step,
        };

        if consumed > budget {
            EnforcementResult::Exceeded(EnforcementAction::FuelExhausted {
                operation: operation.to_string(),
                consumed,
                budget,
            })
        } else {
            EnforcementResult::Allowed
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Get fuel budget for an operation.
    pub fn fuel_budget_for(&self, operation: &str) -> u64 {
        match operation {
            "step" => self.policy.fuel.per_step,
            "reset" => self.policy.fuel.per_reset,
            "batch" => self.policy.fuel.per_batch,
            "init" => self.policy.fuel.per_init,
            "snapshot" => self.policy.fuel.per_snapshot,
            "restore" => self.policy.fuel.per_restore,
            _ => self.policy.fuel.per_step,
        }
    }

    /// Get timeout for an operation.
    pub fn timeout_for(&self, operation: &str) -> Duration {
        match operation {
            "step" => self.policy.timeout.step_duration(),
            "reset" => self.policy.timeout.reset_duration(),
            "batch" => self.policy.timeout.batch_duration(),
            "init" => Duration::from_millis(self.policy.timeout.init_ms),
            "snapshot" | "restore" => Duration::from_millis(self.policy.timeout.snapshot_ms),
            _ => self.policy.timeout.step_duration(),
        }
    }

    /// Reset all counters (for new instance/episode).
    pub fn reset_counters(&self) {
        self.fuel_consumed.store(0, Ordering::Relaxed);
    }

    /// Create a guard for timed operations.
    pub fn timed_guard(&self, operation: &str) -> TimedGuard {
        TimedGuard {
            start: Instant::now(),
            timeout: self.timeout_for(operation),
            operation: operation.to_string(),
        }
    }
}

/// Guard for timed operations.
#[derive(Debug)]
pub struct TimedGuard {
    start: Instant,
    timeout: Duration,
    operation: String,
}

impl TimedGuard {
    /// Check if operation has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.start.elapsed() > self.timeout
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get remaining time.
    pub fn remaining(&self) -> Duration {
        self.timeout.saturating_sub(self.start.elapsed())
    }

    /// Finalize and check for timeout.
    pub fn finalize(self) -> EnforcementResult {
        let elapsed = self.start.elapsed();
        if elapsed > self.timeout {
            EnforcementResult::Exceeded(EnforcementAction::Timeout {
                operation: self.operation,
                elapsed,
                limit: self.timeout,
            })
        } else {
            EnforcementResult::Allowed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PolicyBuilder;

    fn test_policy() -> Policy {
        PolicyBuilder::new()
            .max_memory_mb(64)
            .fuel_per_step(1_000_000)
            .timeout_step_ms(100)
            .allow_filesystem_read(&["/data"])
            .deny_network()
            .build()
    }

    #[test]
    fn test_enforcer_creation() {
        let enforcer = BudgetEnforcer::new(test_policy());
        assert_eq!(enforcer.fuel_consumed(), 0);
        assert_eq!(enforcer.steps_executed(), 0);
    }

    #[test]
    fn test_enforcer_record_fuel() {
        let enforcer = BudgetEnforcer::new(test_policy());
        enforcer.record_fuel(500_000);
        assert_eq!(enforcer.fuel_consumed(), 500_000);
        enforcer.record_fuel(300_000);
        assert_eq!(enforcer.fuel_consumed(), 800_000);
    }

    #[test]
    fn test_enforcer_record_steps() {
        let enforcer = BudgetEnforcer::new(test_policy());
        enforcer.record_step();
        enforcer.record_step();
        assert_eq!(enforcer.steps_executed(), 2);
    }

    #[test]
    fn test_enforcer_check_memory_allowed() {
        let enforcer = BudgetEnforcer::new(test_policy());
        let result = enforcer.check_memory(32 * 1024 * 1024); // 32 MB
        assert!(result.is_allowed());
    }

    #[test]
    fn test_enforcer_check_memory_exceeded() {
        let enforcer = BudgetEnforcer::new(test_policy());
        let result = enforcer.check_memory(128 * 1024 * 1024); // 128 MB > 64 MB limit
        assert!(result.is_denied());
    }

    #[test]
    fn test_enforcer_check_read_allowed() {
        let enforcer = BudgetEnforcer::new(test_policy());
        let result = enforcer.check_read("/data/file.txt");
        assert!(result.is_allowed());
    }

    #[test]
    fn test_enforcer_check_read_denied() {
        let enforcer = BudgetEnforcer::new(test_policy());
        let result = enforcer.check_read("/etc/passwd");
        assert!(result.is_denied());
    }

    #[test]
    fn test_enforcer_check_network_denied() {
        let enforcer = BudgetEnforcer::new(test_policy());
        let result = enforcer.check_network();
        assert!(result.is_denied());
    }

    #[test]
    fn test_enforcer_check_network_allowed() {
        let policy = PolicyBuilder::new().allow_network().build();
        let enforcer = BudgetEnforcer::new(policy);
        let result = enforcer.check_network();
        assert!(result.is_allowed());
    }

    #[test]
    fn test_enforcer_fuel_budget_exceeded() {
        let enforcer = BudgetEnforcer::new(test_policy());
        enforcer.record_fuel(1_500_000); // Exceeds 1_000_000 per_step
        let result = enforcer.check_fuel("step");
        assert!(matches!(result, EnforcementResult::Exceeded(_)));
    }

    #[test]
    fn test_enforcer_fuel_budget_within() {
        let enforcer = BudgetEnforcer::new(test_policy());
        enforcer.record_fuel(500_000);
        let result = enforcer.check_fuel("step");
        assert!(result.is_allowed());
    }

    #[test]
    fn test_enforcer_reset_counters() {
        let enforcer = BudgetEnforcer::new(test_policy());
        enforcer.record_fuel(500_000);
        enforcer.reset_counters();
        assert_eq!(enforcer.fuel_consumed(), 0);
    }

    #[test]
    fn test_timed_guard() {
        let policy = PolicyBuilder::new().timeout_step_ms(1000).build();
        let enforcer = BudgetEnforcer::new(policy);
        let guard = enforcer.timed_guard("step");
        assert!(!guard.is_timed_out());
        assert!(guard.remaining() > Duration::from_millis(900));
    }

    #[test]
    fn test_enforcement_action_to_error() {
        let action = EnforcementAction::FuelExhausted {
            operation: "step".to_string(),
            consumed: 1_500_000,
            budget: 1_000_000,
        };
        let error = action.to_error();
        assert!(error.is_budget_error());
    }

    #[test]
    fn test_enforcement_result_methods() {
        let result = EnforcementResult::Allowed;
        assert!(result.is_allowed());
        assert!(!result.is_denied());
        assert!(result.action().is_none());

        let result = EnforcementResult::Denied(EnforcementAction::CapabilityDenied {
            capability: "network".to_string(),
            reason: "Denied".to_string(),
        });
        assert!(!result.is_allowed());
        assert!(result.is_denied());
        assert!(result.action().is_some());
    }

    #[test]
    fn test_fuel_budget_for() {
        let enforcer = BudgetEnforcer::new(test_policy());
        assert_eq!(enforcer.fuel_budget_for("step"), 1_000_000);
        assert_eq!(enforcer.fuel_budget_for("reset"), 5_000_000);
    }

    #[test]
    fn test_timeout_for() {
        let enforcer = BudgetEnforcer::new(test_policy());
        assert_eq!(enforcer.timeout_for("step"), Duration::from_millis(100));
    }
}
