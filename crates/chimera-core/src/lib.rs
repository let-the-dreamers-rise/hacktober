//! # chimera-core
//!
//! Language-agnostic building blocks for the Chimera interpreter. Nothing in
//! here knows about syntax or values; it is the small set of primitives the
//! rest of the workspace leans on:
//!
//! * [`rng`] – a fast, seedable, *reproducible* PRNG (xoshiro256\*\*). Backs
//!   `random`, `seed` and weight initialisation, so a seeded program prints
//!   the same numbers on every platform.
//! * [`checksum`] – CRC-32, Adler-32 and FNV-1a. FNV-1a drives the feature
//!   hashing of the offline text embeddings; all three are also builtins.
//! * [`codec`] – a compact binary encoding (fixed-width big-endian integers,
//!   LEB128 varints, length-prefixed bytes) with `Encode`/`Decode` traits,
//!   groundwork for the planned bytecode cache.
//! * [`diagnostic`] – source spans and rustc-style error rendering.
//! * [`text`] – edit distance, "did you mean" suggestions, string escaping
//!   and human-readable units.
//! * [`table`] – box-drawn tables for query results and model summaries.
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
