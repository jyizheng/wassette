// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Session management for environment instances.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::SessionConfig;
use crate::error::{BridgeError, BridgeResult};

/// Unique identifier for an environment session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Create a new unique session ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("session-{:08x}", id))
    }

    /// Create a session ID from a string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the session ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// State of an environment session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Session is created but not yet reset.
    Created,
    /// Session is ready (has been reset).
    Ready,
    /// Session is currently executing a step.
    Stepping,
    /// Session has terminated (episode done).
    Terminated,
    /// Session encountered an error.
    Error,
    /// Session is closed.
    Closed,
}

impl SessionState {
    /// Check if the session can accept a step action.
    pub fn can_step(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if the session can be reset.
    pub fn can_reset(&self) -> bool {
        matches!(
            self,
            Self::Created | Self::Ready | Self::Terminated | Self::Error
        )
    }

    /// Check if the session is active (not closed).
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Closed)
    }
}

/// An environment session representing a single instance.
#[derive(Debug)]
pub struct EnvSession {
    /// Unique session identifier.
    pub id: SessionId,

    /// Session configuration.
    pub config: SessionConfig,

    /// Current session state.
    pub state: SessionState,

    /// Total steps executed in this session.
    pub total_steps: u64,

    /// Steps in current episode.
    pub episode_steps: u64,

    /// Total episodes (resets) in this session.
    pub total_episodes: u64,

    /// When the session was created.
    pub created_at: Instant,

    /// When the session was last active.
    pub last_active: Instant,

    /// Last observation (for debugging).
    pub last_observation: Option<serde_json::Value>,

    /// Cumulative reward for current episode.
    pub episode_reward: f64,

    /// Internal instance handle (opaque).
    instance_handle: Option<u64>,
}

impl EnvSession {
    /// Create a new session with the given configuration.
    pub fn new(config: SessionConfig) -> Self {
        let now = Instant::now();
        Self {
            id: SessionId::new(),
            config,
            state: SessionState::Created,
            total_steps: 0,
            episode_steps: 0,
            total_episodes: 0,
            created_at: now,
            last_active: now,
            last_observation: None,
            episode_reward: 0.0,
            instance_handle: None,
        }
    }

    /// Create a session with a specific ID.
    pub fn with_id(id: SessionId, config: SessionConfig) -> Self {
        let mut session = Self::new(config);
        session.id = id;
        session
    }

    /// Set the internal instance handle.
    pub fn set_instance_handle(&mut self, handle: u64) {
        self.instance_handle = Some(handle);
    }

    /// Get the internal instance handle.
    pub fn instance_handle(&self) -> Option<u64> {
        self.instance_handle
    }

    /// Mark the session as reset.
    pub fn mark_reset(&mut self, observation: serde_json::Value) {
        self.state = SessionState::Ready;
        self.episode_steps = 0;
        self.episode_reward = 0.0;
        self.total_episodes += 1;
        self.last_observation = Some(observation);
        self.last_active = Instant::now();
    }

    /// Mark the session as stepped.
    pub fn mark_step(&mut self, observation: serde_json::Value, reward: f64, terminated: bool) {
        self.total_steps += 1;
        self.episode_steps += 1;
        self.episode_reward += reward;
        self.last_observation = Some(observation);
        self.last_active = Instant::now();

        if terminated {
            self.state = SessionState::Terminated;
        } else {
            self.state = SessionState::Ready;
        }
    }

    /// Mark the session as having an error.
    pub fn mark_error(&mut self) {
        self.state = SessionState::Error;
        self.last_active = Instant::now();
    }

    /// Mark the session as closed.
    pub fn mark_closed(&mut self) {
        self.state = SessionState::Closed;
        self.last_active = Instant::now();
    }

    /// Get session duration since creation.
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last activity.
    pub fn idle_time(&self) -> Duration {
        self.last_active.elapsed()
    }

    /// Get session info as JSON.
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({
            "session_id": self.id.as_str(),
            "state": format!("{:?}", self.state),
            "total_steps": self.total_steps,
            "episode_steps": self.episode_steps,
            "total_episodes": self.total_episodes,
            "episode_reward": self.episode_reward,
            "duration_secs": self.duration().as_secs_f64(),
            "idle_secs": self.idle_time().as_secs_f64(),
        })
    }
}

/// Manager for multiple environment sessions.
#[derive(Debug)]
pub struct SessionManager {
    sessions: HashMap<SessionId, EnvSession>,
    max_sessions: usize,
    total_created: u64,
    total_closed: u64,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: HashMap::new(),
            max_sessions,
            total_created: 0,
            total_closed: 0,
        }
    }

    /// Create a new session.
    pub fn create_session(&mut self, config: SessionConfig) -> BridgeResult<SessionId> {
        if self.active_count() >= self.max_sessions {
            return Err(BridgeError::max_sessions_exceeded(self.max_sessions));
        }

        let session = EnvSession::new(config);
        let id = session.id.clone();
        self.sessions.insert(id.clone(), session);
        self.total_created += 1;

        Ok(id)
    }

    /// Get a session by ID.
    pub fn get(&self, id: &SessionId) -> BridgeResult<&EnvSession> {
        self.sessions
            .get(id)
            .ok_or_else(|| BridgeError::session_not_found(id.as_str()))
    }

    /// Get a mutable reference to a session.
    pub fn get_mut(&mut self, id: &SessionId) -> BridgeResult<&mut EnvSession> {
        self.sessions
            .get_mut(id)
            .ok_or_else(|| BridgeError::session_not_found(id.as_str()))
    }

    /// Close a session.
    pub fn close_session(&mut self, id: &SessionId) -> BridgeResult<()> {
        if let Some(session) = self.sessions.get_mut(id) {
            session.mark_closed();
            self.total_closed += 1;
            Ok(())
        } else {
            Err(BridgeError::session_not_found(id.as_str()))
        }
    }

    /// Remove a closed session.
    pub fn remove_session(&mut self, id: &SessionId) -> BridgeResult<EnvSession> {
        self.sessions
            .remove(id)
            .ok_or_else(|| BridgeError::session_not_found(id.as_str()))
    }

    /// List all active sessions.
    pub fn list_sessions(&self) -> Vec<&EnvSession> {
        self.sessions
            .values()
            .filter(|s| s.state.is_active())
            .collect()
    }

    /// Get the number of active sessions.
    pub fn active_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state.is_active())
            .count()
    }

    /// Get manager statistics.
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "active_sessions": self.active_count(),
            "max_sessions": self.max_sessions,
            "total_created": self.total_created,
            "total_closed": self.total_closed,
        })
    }

    /// Clean up idle sessions (older than timeout).
    pub fn cleanup_idle(&mut self, idle_timeout: Duration) -> Vec<SessionId> {
        let idle_ids: Vec<SessionId> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.idle_time() > idle_timeout)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &idle_ids {
            if let Some(session) = self.sessions.get_mut(id) {
                session.mark_closed();
                self.total_closed += 1;
            }
        }

        idle_ids
    }
}

/// Thread-safe wrapper for SessionManager.
#[derive(Debug, Clone)]
pub struct SharedSessionManager(Arc<Mutex<SessionManager>>);

impl SharedSessionManager {
    /// Create a new shared session manager.
    pub fn new(max_sessions: usize) -> Self {
        Self(Arc::new(Mutex::new(SessionManager::new(max_sessions))))
    }

    /// Create a new session.
    pub fn create_session(&self, config: SessionConfig) -> BridgeResult<SessionId> {
        self.0.lock().unwrap().create_session(config)
    }

    /// Execute an operation on a session.
    pub fn with_session<F, R>(&self, id: &SessionId, f: F) -> BridgeResult<R>
    where
        F: FnOnce(&mut EnvSession) -> R,
    {
        let mut manager = self.0.lock().unwrap();
        let session = manager.get_mut(id)?;
        Ok(f(session))
    }

    /// Close a session.
    pub fn close_session(&self, id: &SessionId) -> BridgeResult<()> {
        self.0.lock().unwrap().close_session(id)
    }

    /// Get manager statistics.
    pub fn stats(&self) -> serde_json::Value {
        self.0.lock().unwrap().stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_unique() {
        let id1 = SessionId::new();
        let id2 = SessionId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_session_id_format() {
        let id = SessionId::new();
        assert!(id.as_str().starts_with("session-"));
    }

    #[test]
    fn test_session_state_transitions() {
        assert!(SessionState::Created.can_reset());
        assert!(SessionState::Ready.can_step());
        assert!(SessionState::Ready.can_reset());
        assert!(!SessionState::Stepping.can_step());
        assert!(SessionState::Terminated.can_reset());
        assert!(!SessionState::Closed.is_active());
    }

    #[test]
    fn test_env_session_new() {
        let config = SessionConfig::new().with_seed(42);
        let session = EnvSession::new(config);

        assert_eq!(session.state, SessionState::Created);
        assert_eq!(session.total_steps, 0);
        assert_eq!(session.total_episodes, 0);
    }

    #[test]
    fn test_env_session_reset() {
        let mut session = EnvSession::new(SessionConfig::new());
        session.mark_reset(serde_json::json!([1, 2, 3]));

        assert_eq!(session.state, SessionState::Ready);
        assert_eq!(session.total_episodes, 1);
        assert_eq!(session.episode_steps, 0);
    }

    #[test]
    fn test_env_session_step() {
        let mut session = EnvSession::new(SessionConfig::new());
        session.mark_reset(serde_json::json!([0]));
        session.mark_step(serde_json::json!([1]), 1.0, false);

        assert_eq!(session.state, SessionState::Ready);
        assert_eq!(session.total_steps, 1);
        assert_eq!(session.episode_steps, 1);
        assert_eq!(session.episode_reward, 1.0);
    }

    #[test]
    fn test_env_session_terminate() {
        let mut session = EnvSession::new(SessionConfig::new());
        session.mark_reset(serde_json::json!([0]));
        session.mark_step(serde_json::json!([1]), 10.0, true);

        assert_eq!(session.state, SessionState::Terminated);
    }

    #[test]
    fn test_session_manager_create() {
        let mut manager = SessionManager::new(4);
        let id = manager.create_session(SessionConfig::new()).unwrap();

        assert!(manager.get(&id).is_ok());
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_session_manager_max_sessions() {
        let mut manager = SessionManager::new(2);
        manager.create_session(SessionConfig::new()).unwrap();
        manager.create_session(SessionConfig::new()).unwrap();

        let result = manager.create_session(SessionConfig::new());
        assert!(matches!(
            result,
            Err(BridgeError::MaxSessionsExceeded { .. })
        ));
    }

    #[test]
    fn test_session_manager_close() {
        let mut manager = SessionManager::new(4);
        let id = manager.create_session(SessionConfig::new()).unwrap();

        manager.close_session(&id).unwrap();
        let session = manager.get(&id).unwrap();
        assert_eq!(session.state, SessionState::Closed);
    }

    #[test]
    fn test_session_manager_not_found() {
        let manager = SessionManager::new(4);
        let fake_id = SessionId::from_string("nonexistent");

        assert!(matches!(
            manager.get(&fake_id),
            Err(BridgeError::SessionNotFound { .. })
        ));
    }

    #[test]
    fn test_shared_session_manager() {
        let manager = SharedSessionManager::new(4);
        let id = manager.create_session(SessionConfig::new()).unwrap();

        let info = manager.with_session(&id, |session| session.info()).unwrap();
        assert!(info.get("session_id").is_some());
    }

    #[test]
    fn test_session_info_json() {
        let session = EnvSession::new(SessionConfig::new());
        let info = session.info();

        assert!(info.get("session_id").is_some());
        assert!(info.get("state").is_some());
        assert!(info.get("total_steps").is_some());
    }
}
