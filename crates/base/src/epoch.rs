use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// A shared epoch counter for pipeline cancellation.
///
/// All nodes in a pipeline share the same Epoch. When `advance()` is called,
/// every node independently notices the epoch changed and stops processing
/// stale data.
#[derive(Clone)]
pub struct Epoch {
    value: Arc<AtomicU64>,
}

impl Epoch {
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Get the current epoch value.
    pub fn current(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Advance to the next epoch. Returns the new epoch value.
    /// This instantly invalidates all in-flight data from previous epochs.
    pub fn advance(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Check if the given epoch is still current.
    pub fn is_current(&self, epoch: u64) -> bool {
        epoch == self.current()
    }
}

/// A value stamped with the epoch it was created in.
#[derive(Clone, Debug)]
pub struct Stamped<T> {
    pub epoch: u64,
    pub inner: T,
}
