// Modified from sp1-hypercube 6.4.0 by the sp1-metal project; see CHANGES.md.
use std::{sync::Arc, time::Duration};

use thiserror::Error;
use tokio::sync::{AcquireError, OwnedSemaphorePermit, Semaphore};
use tracing::Span;

/// A permit for the prover.
#[derive(Debug)]
pub struct ProverPermit {
    /// The underlying permit.
    #[allow(dead_code)]
    permit: OwnedSemaphorePermit,
    /// The span for the permit lifetime.
    span: Span,
    /// The time the permit was acquired.
    time: tokio::time::Instant,
}

impl ProverPermit {
    /// Release the permit and return the duration it was held for.
    #[must_use]
    pub fn release(self) -> Duration {
        self.time.elapsed()
    }
}

impl Drop for ProverPermit {
    fn drop(&mut self) {
        let duration = self.time.elapsed();
        tracing::debug!(parent: &self.span, "permit acquired for {:?} ", duration);
    }
}

/// A semaphore for the prover.
#[derive(Debug, Clone)]
pub struct ProverSemaphore {
    /// The underlying semaphore.
    sem: Arc<Semaphore>,
}

impl ProverSemaphore {
    /// Create a new prover semaphore with the given number of permits.
    ///
    /// `SP1_METAL_PERMITS` raises a cap of more than one, which is the one that
    /// decides how many shards prove at once. `sp1-prover` hardcodes it to four
    /// -- `cpu_worker_builder_with_machine` builds the semaphore itself and the
    /// SDK offers no way past it -- and four shards keep this machine about
    /// half busy, so whether more of them would fill it is a question nothing
    /// else can ask. The caps of one, which serialise setup and recursion, are
    /// left alone.
    #[must_use]
    #[inline]
    pub fn new(max_permits: usize) -> Self {
        let max_permits = if max_permits > 1 {
            std::env::var("SP1_METAL_PERMITS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .unwrap_or(max_permits)
        } else {
            max_permits
        };
        Self { sem: Arc::new(Semaphore::new(max_permits)) }
    }

    /// Acquire a permit.
    #[inline]
    pub async fn acquire(self) -> Result<ProverPermit, ProverAcquireError> {
        let span = tracing::Span::current();
        let permit = self.sem.acquire_owned().await?;
        let time = tokio::time::Instant::now();
        Ok(ProverPermit { permit, span, time })
    }

    /// Acquire multiple permits.
    #[inline]
    pub async fn acquire_many(self, n: u32) -> Result<ProverPermit, ProverAcquireError> {
        let span = tracing::Span::current();
        let permit = self.sem.acquire_many_owned(n).await?;
        let time = tokio::time::Instant::now();
        Ok(ProverPermit { permit, span, time })
    }
}

/// An error that occurs when acquiring a permit.
#[derive(Debug, Error)]
#[error("failed to acquire permit")]
pub struct ProverAcquireError(#[from] AcquireError);
