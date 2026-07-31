// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Telemetry collection for policy enforcement.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Events that can be recorded by telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyEvent {
    /// Budget overrun occurred.
    BudgetOverrun(BudgetOverrun),
    /// Capability was denied.
    CapabilityDenial(CapabilityDenial),
    /// Trap/crash occurred.
    Trap(TrapInfo),
    /// Step completed with timing.
    StepCompleted {
        /// Time taken.
        duration_us: u64,
        /// Fuel consumed.
        fuel_consumed: u64,
    },
    /// Reset completed with timing.
    ResetCompleted {
        /// Time taken.
        duration_us: u64,
        /// Fuel consumed.
        fuel_consumed: u64,
    },
}

/// Information about a budget overrun.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetOverrun {
    /// Type of budget that was exceeded.
    pub budget_type: BudgetType,
    /// Limit that was set.
    pub limit: u64,
    /// Actual value that exceeded the limit.
    pub actual: u64,
    /// Operation during which overrun occurred.
    pub operation: String,
    /// Timestamp (microseconds since telemetry start).
    pub timestamp_us: u64,
}

/// Types of budgets that can be exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetType {
    /// Fuel/instruction budget.
    Fuel,
    /// Memory budget.
    Memory,
    /// Timeout budget.
    Timeout,
}

impl std::fmt::Display for BudgetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fuel => write!(f, "fuel"),
            Self::Memory => write!(f, "memory"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

/// Information about a capability denial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityDenial {
    /// Capability that was denied.
    pub capability: String,
    /// Reason for denial.
    pub reason: String,
    /// Additional context (e.g., path that was denied).
    pub context: Option<String>,
    /// Timestamp (microseconds since telemetry start).
    pub timestamp_us: u64,
}

/// Information about a trap/crash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrapInfo {
    /// Trap code or type.
    pub trap_code: String,
    /// Trap message.
    pub message: String,
    /// Function where trap occurred (if known).
    pub function: Option<String>,
    /// Timestamp (microseconds since telemetry start).
    pub timestamp_us: u64,
}

impl TrapInfo {
    /// Create a new trap info.
    pub fn new(trap_code: &str, message: &str) -> Self {
        Self {
            trap_code: trap_code.to_string(),
            message: message.to_string(),
            function: None,
            timestamp_us: 0,
        }
    }

    /// Set the function name.
    pub fn with_function(mut self, function: &str) -> Self {
        self.function = Some(function.to_string());
        self
    }
}

/// Collected telemetry data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyTelemetry {
    /// Total steps executed.
    pub steps_total: u64,
    /// Total resets executed.
    pub resets_total: u64,
    /// Total fuel consumed.
    pub fuel_consumed_total: u64,
    /// Budget overruns by type.
    pub budget_overruns: HashMap<String, u64>,
    /// Capability denials by capability.
    pub capability_denials: HashMap<String, u64>,
    /// Trap count by trap code.
    pub traps: HashMap<String, u64>,
    /// Step timing statistics (microseconds).
    pub step_timing: TimingStats,
    /// Reset timing statistics (microseconds).
    pub reset_timing: TimingStats,
    /// Collection duration.
    pub collection_duration_ms: u64,
}

impl Default for PolicyTelemetry {
    fn default() -> Self {
        Self {
            steps_total: 0,
            resets_total: 0,
            fuel_consumed_total: 0,
            budget_overruns: HashMap::new(),
            capability_denials: HashMap::new(),
            traps: HashMap::new(),
            step_timing: TimingStats::default(),
            reset_timing: TimingStats::default(),
            collection_duration_ms: 0,
        }
    }
}

impl PolicyTelemetry {
    /// Calculate budget overrun rate.
    pub fn overrun_rate(&self) -> f64 {
        let total_ops = self.steps_total + self.resets_total;
        if total_ops == 0 {
            return 0.0;
        }
        let total_overruns: u64 = self.budget_overruns.values().sum();
        total_overruns as f64 / total_ops as f64
    }

    /// Calculate trap rate.
    pub fn trap_rate(&self) -> f64 {
        let total_ops = self.steps_total + self.resets_total;
        if total_ops == 0 {
            return 0.0;
        }
        let total_traps: u64 = self.traps.values().sum();
        total_traps as f64 / total_ops as f64
    }
}

/// Timing statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingStats {
    /// Number of samples.
    pub count: u64,
    /// Sum of all timings (microseconds).
    pub sum_us: u64,
    /// Minimum timing (microseconds).
    pub min_us: u64,
    /// Maximum timing (microseconds).
    pub max_us: u64,
}

impl TimingStats {
    /// Add a timing sample.
    pub fn record(&mut self, duration_us: u64) {
        self.count += 1;
        self.sum_us += duration_us;
        if self.count == 1 {
            self.min_us = duration_us;
            self.max_us = duration_us;
        } else {
            self.min_us = self.min_us.min(duration_us);
            self.max_us = self.max_us.max(duration_us);
        }
    }

    /// Calculate mean timing.
    pub fn mean_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_us as f64 / self.count as f64
        }
    }
}

/// Collector for policy telemetry.
#[derive(Debug)]
pub struct TelemetryCollector {
    /// Start time.
    start_time: Instant,
    /// Step count.
    steps: AtomicU64,
    /// Reset count.
    resets: AtomicU64,
    /// Total fuel consumed.
    fuel_consumed: AtomicU64,
    /// Budget overruns.
    budget_overruns: Mutex<HashMap<BudgetType, u64>>,
    /// Capability denials.
    capability_denials: Mutex<HashMap<String, u64>>,
    /// Trap counts.
    traps: Mutex<HashMap<String, u64>>,
    /// Step timings.
    step_timings: Mutex<Vec<u64>>,
    /// Reset timings.
    reset_timings: Mutex<Vec<u64>>,
    /// Max events to store.
    max_events: usize,
    /// Recorded events.
    events: Mutex<Vec<PolicyEvent>>,
}

impl TelemetryCollector {
    /// Create a new telemetry collector.
    pub fn new() -> Self {
        Self::with_max_events(10_000)
    }

    /// Create with custom max events.
    pub fn with_max_events(max_events: usize) -> Self {
        Self {
            start_time: Instant::now(),
            steps: AtomicU64::new(0),
            resets: AtomicU64::new(0),
            fuel_consumed: AtomicU64::new(0),
            budget_overruns: Mutex::new(HashMap::new()),
            capability_denials: Mutex::new(HashMap::new()),
            traps: Mutex::new(HashMap::new()),
            step_timings: Mutex::new(Vec::new()),
            reset_timings: Mutex::new(Vec::new()),
            max_events,
            events: Mutex::new(Vec::new()),
        }
    }

    /// Create a shared collector.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Get elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get timestamp in microseconds.
    fn timestamp_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    /// Record a step completion.
    pub fn record_step(&self, duration: Duration, fuel_consumed: u64) {
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.fuel_consumed
            .fetch_add(fuel_consumed, Ordering::Relaxed);

        let duration_us = duration.as_micros() as u64;
        if let Ok(mut timings) = self.step_timings.lock() {
            if timings.len() < self.max_events {
                timings.push(duration_us);
            }
        }

        self.record_event(PolicyEvent::StepCompleted {
            duration_us,
            fuel_consumed,
        });
    }

    /// Record a reset completion.
    pub fn record_reset(&self, duration: Duration, fuel_consumed: u64) {
        self.resets.fetch_add(1, Ordering::Relaxed);
        self.fuel_consumed
            .fetch_add(fuel_consumed, Ordering::Relaxed);

        let duration_us = duration.as_micros() as u64;
        if let Ok(mut timings) = self.reset_timings.lock() {
            if timings.len() < self.max_events {
                timings.push(duration_us);
            }
        }

        self.record_event(PolicyEvent::ResetCompleted {
            duration_us,
            fuel_consumed,
        });
    }

    /// Record a budget overrun.
    pub fn record_budget_overrun(
        &self,
        budget_type: BudgetType,
        limit: u64,
        actual: u64,
        operation: &str,
    ) {
        if let Ok(mut overruns) = self.budget_overruns.lock() {
            *overruns.entry(budget_type).or_insert(0) += 1;
        }

        let overrun = BudgetOverrun {
            budget_type,
            limit,
            actual,
            operation: operation.to_string(),
            timestamp_us: self.timestamp_us(),
        };
        self.record_event(PolicyEvent::BudgetOverrun(overrun));
    }

    /// Record a capability denial.
    pub fn record_capability_denial(&self, capability: &str, reason: &str, context: Option<&str>) {
        if let Ok(mut denials) = self.capability_denials.lock() {
            *denials.entry(capability.to_string()).or_insert(0) += 1;
        }

        let denial = CapabilityDenial {
            capability: capability.to_string(),
            reason: reason.to_string(),
            context: context.map(String::from),
            timestamp_us: self.timestamp_us(),
        };
        self.record_event(PolicyEvent::CapabilityDenial(denial));
    }

    /// Record a trap.
    pub fn record_trap(&self, trap_code: &str, message: &str, function: Option<&str>) {
        if let Ok(mut traps) = self.traps.lock() {
            *traps.entry(trap_code.to_string()).or_insert(0) += 1;
        }

        let trap = TrapInfo {
            trap_code: trap_code.to_string(),
            message: message.to_string(),
            function: function.map(String::from),
            timestamp_us: self.timestamp_us(),
        };
        self.record_event(PolicyEvent::Trap(trap));
    }

    /// Record a generic event.
    fn record_event(&self, event: PolicyEvent) {
        if let Ok(mut events) = self.events.lock() {
            if events.len() < self.max_events {
                events.push(event);
            }
        }
    }

    /// Get current telemetry snapshot.
    pub fn snapshot(&self) -> PolicyTelemetry {
        let budget_overruns = self
            .budget_overruns
            .lock()
            .map(|m| m.iter().map(|(k, v)| (k.to_string(), *v)).collect())
            .unwrap_or_default();

        let capability_denials = self
            .capability_denials
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default();

        let traps = self.traps.lock().map(|m| m.clone()).unwrap_or_default();

        let step_timing = self
            .step_timings
            .lock()
            .map(|t| {
                let mut stats = TimingStats::default();
                for &timing in t.iter() {
                    stats.record(timing);
                }
                stats
            })
            .unwrap_or_default();

        let reset_timing = self
            .reset_timings
            .lock()
            .map(|t| {
                let mut stats = TimingStats::default();
                for &timing in t.iter() {
                    stats.record(timing);
                }
                stats
            })
            .unwrap_or_default();

        PolicyTelemetry {
            steps_total: self.steps.load(Ordering::Relaxed),
            resets_total: self.resets.load(Ordering::Relaxed),
            fuel_consumed_total: self.fuel_consumed.load(Ordering::Relaxed),
            budget_overruns,
            capability_denials,
            traps,
            step_timing,
            reset_timing,
            collection_duration_ms: self.elapsed().as_millis() as u64,
        }
    }

    /// Get all recorded events.
    pub fn events(&self) -> Vec<PolicyEvent> {
        self.events.lock().map(|e| e.clone()).unwrap_or_default()
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.steps.store(0, Ordering::Relaxed);
        self.resets.store(0, Ordering::Relaxed);
        self.fuel_consumed.store(0, Ordering::Relaxed);
        if let Ok(mut m) = self.budget_overruns.lock() {
            m.clear();
        }
        if let Ok(mut m) = self.capability_denials.lock() {
            m.clear();
        }
        if let Ok(mut m) = self.traps.lock() {
            m.clear();
        }
        if let Ok(mut v) = self.step_timings.lock() {
            v.clear();
        }
        if let Ok(mut v) = self.reset_timings.lock() {
            v.clear();
        }
        if let Ok(mut v) = self.events.lock() {
            v.clear();
        }
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_collector_creation() {
        let collector = TelemetryCollector::new();
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.steps_total, 0);
        assert_eq!(snapshot.resets_total, 0);
    }

    #[test]
    fn test_record_step() {
        let collector = TelemetryCollector::new();
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_step(Duration::from_micros(150), 60_000);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.steps_total, 2);
        assert_eq!(snapshot.fuel_consumed_total, 110_000);
        assert_eq!(snapshot.step_timing.count, 2);
    }

    #[test]
    fn test_record_reset() {
        let collector = TelemetryCollector::new();
        collector.record_reset(Duration::from_millis(10), 100_000);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.resets_total, 1);
        assert_eq!(snapshot.reset_timing.count, 1);
    }

    #[test]
    fn test_record_budget_overrun() {
        let collector = TelemetryCollector::new();
        collector.record_budget_overrun(BudgetType::Fuel, 1_000_000, 1_500_000, "step");

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.budget_overruns.get("fuel"), Some(&1));
    }

    #[test]
    fn test_record_capability_denial() {
        let collector = TelemetryCollector::new();
        collector.record_capability_denial("network", "Denied by policy", None);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.capability_denials.get("network"), Some(&1));
    }

    #[test]
    fn test_record_trap() {
        let collector = TelemetryCollector::new();
        collector.record_trap("unreachable", "unreachable code reached", Some("step"));

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.traps.get("unreachable"), Some(&1));
    }

    #[test]
    fn test_timing_stats() {
        let mut stats = TimingStats::default();
        stats.record(100);
        stats.record(200);
        stats.record(300);

        assert_eq!(stats.count, 3);
        assert_eq!(stats.sum_us, 600);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 300);
        assert!((stats.mean_us() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_overrun_rate() {
        let collector = TelemetryCollector::new();
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_budget_overrun(BudgetType::Fuel, 1_000_000, 1_500_000, "step");

        let snapshot = collector.snapshot();
        // 1 overrun / 2 steps = 0.5
        assert!((snapshot.overrun_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_trap_rate() {
        let collector = TelemetryCollector::new();
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_trap("unreachable", "trap", None);

        let snapshot = collector.snapshot();
        // 1 trap / 4 steps = 0.25
        assert!((snapshot.trap_rate() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_reset_collector() {
        let collector = TelemetryCollector::new();
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.reset();

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.steps_total, 0);
    }

    #[test]
    fn test_budget_type_display() {
        assert_eq!(BudgetType::Fuel.to_string(), "fuel");
        assert_eq!(BudgetType::Memory.to_string(), "memory");
        assert_eq!(BudgetType::Timeout.to_string(), "timeout");
    }

    #[test]
    fn test_trap_info_builder() {
        let trap = TrapInfo::new("oom", "out of memory").with_function("allocate");
        assert_eq!(trap.trap_code, "oom");
        assert_eq!(trap.function, Some("allocate".to_string()));
    }

    #[test]
    fn test_events_collection() {
        let collector = TelemetryCollector::new();
        collector.record_step(Duration::from_micros(100), 50_000);
        collector.record_trap("trap", "error", None);

        let events = collector.events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_max_events_limit() {
        let collector = TelemetryCollector::with_max_events(5);
        for _ in 0..10 {
            collector.record_step(Duration::from_micros(100), 50_000);
        }

        let events = collector.events();
        assert!(events.len() <= 5);
    }

    #[test]
    fn test_policy_telemetry_serialization() {
        let collector = TelemetryCollector::new();
        collector.record_step(Duration::from_micros(100), 50_000);

        let snapshot = collector.snapshot();
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("steps_total"));

        let parsed: PolicyTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.steps_total, snapshot.steps_total);
    }
}
