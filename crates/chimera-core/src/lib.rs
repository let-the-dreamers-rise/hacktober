//! # chimera-core
//!
//! The foundation every other Chimera subsystem stands on. Nothing in here
//! knows about languages, databases, consensus or light transport; it is the
//! small set of primitives those systems all need and must agree on:
//!
//! * [`rng`] – a fast, seedable, *reproducible* PRNG (xoshiro256\*\*). The Raft
//!   simulator and the path tracer both rely on bit-for-bit determinism.
//! * [`checksum`] – CRC-32, Adler-32 and FNV-1a. Used by the write-ahead log,
//!   the Raft log, snapshots and the PNG encoder.
//! * [`codec`] – a compact binary encoding (fixed-width big-endian integers,
//!   LEB128 varints, length-prefixed bytes) with `Encode`/`Decode` traits.
//! * [`diagnostic`] – source spans and rustc-style error rendering shared by
//!   the Chimera language and the SQL front end.
//! * [`text`] – edit distance, "did you mean" suggestions and human-readable
//!   units.
//! * [`table`] – box-drawn tables for the SQL shell and CLI reports.
//!
//! The crate has zero dependencies and forbids `unsafe`.

pub mod checksum;
pub mod codec;
pub mod diagnostic;
pub mod rng;
pub mod table;
pub mod text;

pub use checksum::{Crc32, adler32, crc32, fnv1a64};
pub use codec::{Decode, DecodeError, Decoder, Encode, Encoder};
pub use diagnostic::{Diagnostic, Label, Severity, SourceFile, Span};
pub use rng::Rng;
pub use table::{Align, Table};
