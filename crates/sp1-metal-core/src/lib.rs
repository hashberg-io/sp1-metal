// SPDX-License-Identifier: MIT OR Apache-2.0
//! The floor of the Metal backend.
//!
//! Two things live here, and they are here rather than in the probe suite for
//! the same reason: the crates that need them sit on both sides of it. The
//! vendored slop crates are dependencies of the probe suite's dependencies, so
//! anything they share with it has to be below both.
//!
//! [`stats`] is process-wide by construction — a counter that only some of the
//! dispatches reach is worse than no counter, because it reads as a measurement
//! — and [`nocopy`] states the one rule the whole design rests on.

// Every crate that dispatches reaches this one, so its surface is the only
// one in the project a stranger has to read. Keep it documented.
#![warn(missing_docs)]

pub mod compile;
pub mod handoff;
pub mod nocopy;
pub mod queue;
pub mod stats;
pub mod tally;
