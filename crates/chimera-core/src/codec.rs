//! A compact, dependency-free binary encoding.
//!
//! Fixed-width integers are big-endian, so encoded unsigned integers sort the
//! same way as their values (handy for database keys). Variable-width integers
//! use LEB128; signed varints are zig-zag encoded first. Byte strings and
//! strings carry a varint length prefix.
//!
//! The [`Encode`] and [`Decode`] traits use the compact varint forms for
//! integer types. Decoding never trusts length prefixes for allocation, so
//! corrupt or hostile input cannot trigger huge allocations.
//!
//! ```
//! use chimera_core::codec::{Decode, Encode};
//!
//! let original = (42u64, String::from("chimera"), vec![Some(-7i64), None]);
//! let bytes = original.to_bytes();
//! let decoded = <(u64, String, Vec<Option<i64>>)>::from_bytes(&bytes).unwrap();
//! assert_eq!(original, decoded);
//! ```

use std::fmt;

/// Why decoding failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The input ended before a complete value was read.
    UnexpectedEof {
        /// Bytes required by the value being read.
        needed: usize,
        /// Bytes actually left.
        remaining: usize,
    },
    /// A varint ran past 10 bytes or overflowed 64 bits.
    VarintOverflow,
    /// A string was not valid UTF-8.
    InvalidUtf8,
    /// An enum discriminant or flag byte had an unknown value.
    InvalidTag {
        /// What was being decoded.
        what: &'static str,
        /// The offending tag.
        tag: u64,
    },
    /// Bytes were left over after a complete value was decoded.
    TrailingBytes(usize),
    /// Any other structural problem.
    Invalid(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEof { needed, remaining } => write!(
                f,
                "unexpected end of input: needed {needed} bytes, {remaining} remaining"
            ),
            DecodeError::VarintOverflow => write!(f, "varint overflows 64 bits"),
            DecodeError::InvalidUtf8 => write!(f, "string is not valid UTF-8"),
            DecodeError::InvalidTag { what, tag } => write!(f, "invalid {what} tag {tag}"),
            DecodeError::TrailingBytes(n) => write!(f, "{n} trailing bytes after value"),
            DecodeError::Invalid(msg) => write!(f, "invalid encoding: {msg}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Appends encoded values to a growable buffer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoder {
    buf: Vec<u8>,
}

impl Encoder {
    /// Creates an empty encoder.
    pub fn new() -> Self {
        Encoder { buf: Vec::new() }
    }

    /// Creates an encoder with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Encoder {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Number of bytes written so far.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The bytes written so far.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Consumes the encoder, returning its buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Discards everything written so far, keeping the allocation.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Writes one byte.
    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Writes a bool as a single 0/1 byte.
    pub fn put_bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    /// Writes a big-endian `u16`.
    pub fn put_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Writes a big-endian `u32`.
    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Writes a big-endian `u64`.
    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Writes a big-endian two's-complement `i64`.
    pub fn put_i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Writes the IEEE-754 bits of an `f64`, big-endian.
    pub fn put_f64(&mut self, v: f64) {
        self.put_u64(v.to_bits());
    }

    /// Writes an unsigned LEB128 varint (1–10 bytes).
    pub fn put_uvarint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }

    /// Writes a zig-zag encoded signed varint.
    pub fn put_ivarint(&mut self, v: i64) {
        self.put_uvarint(((v << 1) ^ (v >> 63)) as u64);
    }

    /// Writes raw bytes with no length prefix.
    pub fn put_raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Writes a varint length followed by the bytes.
    pub fn put_bytes(&mut self, bytes: &[u8]) {
        self.put_uvarint(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }

    /// Writes a varint length followed by UTF-8 bytes.
    pub fn put_str(&mut self, s: &str) {
        self.put_bytes(s.as_bytes());
    }

    /// Writes any [`Encode`] value.
    pub fn put<T: Encode + ?Sized>(&mut self, value: &T) {
        value.encode(self);
    }
}

/// Reads encoded values from a byte slice.
#[derive(Clone, Debug)]
pub struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// Starts decoding at the beginning of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Decoder { buf, pos: 0 }
    }

    /// Bytes consumed so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Whether every byte has been consumed.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Errors if any bytes remain unread.
    pub fn finish(&self) -> Result<(), DecodeError> {
        match self.remaining() {
            0 => Ok(()),
            n => Err(DecodeError::TrailingBytes(n)),
        }
    }

    /// Reads exactly `n` raw bytes.
    pub fn get_raw(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::UnexpectedEof {
                needed: n,
                remaining: self.remaining(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn get_array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.get_raw(N)?);
        Ok(out)
    }

    /// Reads one byte.
    pub fn get_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.get_raw(1)?[0])
    }

    /// Reads a 0/1 bool byte.
    pub fn get_bool(&mut self) -> Result<bool, DecodeError> {
        match self.get_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            tag => Err(DecodeError::InvalidTag {
                what: "bool",
                tag: u64::from(tag),
            }),
        }
    }

    /// Reads a big-endian `u16`.
    pub fn get_u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_be_bytes(self.get_array()?))
    }

    /// Reads a big-endian `u32`.
    pub fn get_u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_be_bytes(self.get_array()?))
    }

    /// Reads a big-endian `u64`.
    pub fn get_u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_be_bytes(self.get_array()?))
    }

    /// Reads a big-endian `i64`.
    pub fn get_i64(&mut self) -> Result<i64, DecodeError> {
        Ok(i64::from_be_bytes(self.get_array()?))
    }

    /// Reads an `f64` from its big-endian IEEE-754 bits.
    pub fn get_f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_bits(self.get_u64()?))
    }

    /// Reads an unsigned LEB128 varint.
    pub fn get_uvarint(&mut self) -> Result<u64, DecodeError> {
        let mut result = 0u64;
        for i in 0..10 {
            let byte = self.get_u8()?;
            let low = u64::from(byte & 0x7F);
            if i == 9 && byte > 1 {
                return Err(DecodeError::VarintOverflow);
            }
            result |= low << (7 * i);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(DecodeError::VarintOverflow)
    }

    /// Reads a zig-zag encoded signed varint.
    pub fn get_ivarint(&mut self) -> Result<i64, DecodeError> {
        let v = self.get_uvarint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    /// Reads a varint length and returns that many bytes.
    pub fn get_bytes(&mut self) -> Result<&'a [u8], DecodeError> {
        let len = self.get_len()?;
        self.get_raw(len)
    }

    /// Reads a length-prefixed UTF-8 string slice.
    pub fn get_str(&mut self) -> Result<&'a str, DecodeError> {
        std::str::from_utf8(self.get_bytes()?).map_err(|_| DecodeError::InvalidUtf8)
    }

    /// Reads a length-prefixed UTF-8 string.
    pub fn get_string(&mut self) -> Result<String, DecodeError> {
        self.get_str().map(str::to_owned)
    }

    /// Reads a varint used as a length or count, rejecting values that cannot
    /// possibly fit in the remaining input.
    pub fn get_len(&mut self) -> Result<usize, DecodeError> {
        let len = self.get_uvarint()?;
        usize::try_from(len)
            .ok()
            .filter(|&n| n <= self.remaining())
            .ok_or(DecodeError::UnexpectedEof {
                needed: usize::try_from(len).unwrap_or(usize::MAX),
                remaining: self.remaining(),
            })
    }

    /// Reads any [`Decode`] value.
    pub fn get<T: Decode>(&mut self) -> Result<T, DecodeError> {
        T::decode(self)
    }
}

/// Types that can be written with an [`Encoder`].
pub trait Encode {
    /// Appends `self` to `enc`.
    fn encode(&self, enc: &mut Encoder);

    /// Encodes `self` into a fresh buffer.
    fn to_bytes(&self) -> Vec<u8> {
        let mut enc = Encoder::new();
        self.encode(&mut enc);
        enc.into_bytes()
    }
}

/// Types that can be read with a [`Decoder`].
pub trait Decode: Sized {
    /// Reads one value from `dec`.
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError>;

    /// Decodes a value that must occupy all of `bytes`.
    fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut dec = Decoder::new(bytes);
        let value = Self::decode(&mut dec)?;
        dec.finish()?;
        Ok(value)
    }
}

impl Encode for u8 {
    fn encode(&self, enc: &mut Encoder) {
        enc.put_u8(*self);
    }
}

impl Decode for u8 {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        dec.get_u8()
    }
}

impl Encode for bool {
    fn encode(&self, enc: &mut Encoder) {
        enc.put_bool(*self);
    }
}

impl Decode for bool {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        dec.get_bool()
    }
}

macro_rules! unsigned_varint {
    ($($t:ty),*) => {$(
        impl Encode for $t {
            fn encode(&self, enc: &mut Encoder) {
                enc.put_uvarint(*self as u64);
            }
        }

        impl Decode for $t {
            fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
                let v = dec.get_uvarint()?;
                <$t>::try_from(v).map_err(|_| {
                    DecodeError::Invalid(format!("{v} out of range for {}", stringify!($t)))
                })
            }
        }
    )*};
}

macro_rules! signed_varint {
    ($($t:ty),*) => {$(
        impl Encode for $t {
            fn encode(&self, enc: &mut Encoder) {
                enc.put_ivarint(*self as i64);
            }
        }

        impl Decode for $t {
            fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
                let v = dec.get_ivarint()?;
                <$t>::try_from(v).map_err(|_| {
                    DecodeError::Invalid(format!("{v} out of range for {}", stringify!($t)))
                })
            }
        }
    )*};
}

unsigned_varint!(u16, u32, u64, usize);
signed_varint!(i16, i32, i64, isize);

impl Encode for f64 {
    fn encode(&self, enc: &mut Encoder) {
        enc.put_f64(*self);
    }
}

impl Decode for f64 {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        dec.get_f64()
    }
}

impl Encode for str {
    fn encode(&self, enc: &mut Encoder) {
        enc.put_str(self);
    }
}

impl Encode for String {
    fn encode(&self, enc: &mut Encoder) {
        enc.put_str(self);
    }
}

impl Decode for String {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        dec.get_string()
    }
}

impl<T: Encode> Encode for [T] {
    fn encode(&self, enc: &mut Encoder) {
        enc.put_uvarint(self.len() as u64);
        for item in self {
            item.encode(enc);
        }
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode(&self, enc: &mut Encoder) {
        self.as_slice().encode(enc);
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        // Every element occupies at least one byte, so `get_len` bounds the
        // allocation by the input size.
        let len = dec.get_len()?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(T::decode(dec)?);
        }
        Ok(out)
    }
}

impl<T: Encode> Encode for Option<T> {
    fn encode(&self, enc: &mut Encoder) {
        match self {
            None => enc.put_u8(0),
            Some(v) => {
                enc.put_u8(1);
                v.encode(enc);
            }
        }
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        match dec.get_u8()? {
            0 => Ok(None),
            1 => Ok(Some(T::decode(dec)?)),
            tag => Err(DecodeError::InvalidTag {
                what: "Option",
                tag: u64::from(tag),
            }),
        }
    }
}

impl<T: Encode + ?Sized> Encode for Box<T> {
    fn encode(&self, enc: &mut Encoder) {
        (**self).encode(enc);
    }
}

impl<T: Decode> Decode for Box<T> {
    fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        T::decode(dec).map(Box::new)
    }
}

impl<T: Encode + ?Sized> Encode for &T {
    fn encode(&self, enc: &mut Encoder) {
        (**self).encode(enc);
    }
}

macro_rules! tuple_codec {
    ($($name:ident),+) => {
        impl<$($name: Encode),+> Encode for ($($name,)+) {
            #[allow(non_snake_case)]
            fn encode(&self, enc: &mut Encoder) {
                let ($($name,)+) = self;
                $($name.encode(enc);)+
            }
        }

        impl<$($name: Decode),+> Decode for ($($name,)+) {
            fn decode(dec: &mut Decoder<'_>) -> Result<Self, DecodeError> {
                Ok(($($name::decode(dec)?,)+))
            }
        }
    };
}

tuple_codec!(A);
tuple_codec!(A, B);
tuple_codec!(A, B, C);
tuple_codec!(A, B, C, D);
tuple_codec!(A, B, C, D, E);

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T: Encode + Decode + PartialEq + fmt::Debug>(value: T) {
        let bytes = value.to_bytes();
        assert_eq!(T::from_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn uvarint_boundaries() {
        for v in [0, 1, 127, 128, 16_383, 16_384, u32::MAX as u64, u64::MAX] {
            let mut enc = Encoder::new();
            enc.put_uvarint(v);
            let mut dec = Decoder::new(enc.as_bytes());
            assert_eq!(dec.get_uvarint().unwrap(), v);
            assert!(dec.is_empty());
        }
        let mut enc = Encoder::new();
        enc.put_uvarint(127);
        assert_eq!(enc.len(), 1);
        enc.clear();
        enc.put_uvarint(128);
        assert_eq!(enc.as_bytes(), &[0x80, 0x01]);
        enc.clear();
        enc.put_uvarint(u64::MAX);
        assert_eq!(enc.len(), 10);
    }

    #[test]
    fn ivarint_zigzag() {
        for v in [0, -1, 1, -64, 64, i64::MIN, i64::MAX] {
            let mut enc = Encoder::new();
            enc.put_ivarint(v);
            assert_eq!(Decoder::new(enc.as_bytes()).get_ivarint().unwrap(), v);
        }
        let mut enc = Encoder::new();
        enc.put_ivarint(-1);
        assert_eq!(enc.as_bytes(), &[1]);
    }

    #[test]
    fn varint_overflow_is_rejected() {
        let bytes = [0xFF; 11];
        assert_eq!(
            Decoder::new(&bytes).get_uvarint(),
            Err(DecodeError::VarintOverflow)
        );
        let mut bytes = [0xFF; 10];
        bytes[9] = 0x02;
        assert_eq!(
            Decoder::new(&bytes).get_uvarint(),
            Err(DecodeError::VarintOverflow)
        );
    }

    #[test]
    fn big_endian_fixed_width_sorts_like_values() {
        let mut encoded: Vec<Vec<u8>> = [5u64, 300, 1, 70_000, 0]
            .iter()
            .map(|&v| {
                let mut enc = Encoder::new();
                enc.put_u64(v);
                enc.into_bytes()
            })
            .collect();
        encoded.sort();
        let decoded: Vec<u64> = encoded
            .iter()
            .map(|b| Decoder::new(b).get_u64().unwrap())
            .collect();
        assert_eq!(decoded, vec![0, 1, 5, 300, 70_000]);
    }

    #[test]
    fn trait_roundtrips() {
        roundtrip(0u8);
        roundtrip(true);
        roundtrip(u64::MAX);
        roundtrip(i64::MIN);
        roundtrip(-12_345i32);
        roundtrip(3.25f64);
        roundtrip(String::from("héllo, chimera 🐉"));
        roundtrip(vec![1u32, 2, 3]);
        roundtrip(vec![0u8, 255, 7]);
        roundtrip(Some(String::from("x")));
        roundtrip(None::<u64>);
        roundtrip((1u64, String::from("a"), vec![Some(2i64), None]));
        roundtrip(Box::new(9u16));
    }

    #[test]
    fn truncated_input_is_an_error() {
        let bytes = (7u64, String::from("abcdef")).to_bytes();
        for cut in 0..bytes.len() {
            assert!(<(u64, String)>::from_bytes(&bytes[..cut]).is_err());
        }
    }

    #[test]
    fn trailing_bytes_are_an_error() {
        let mut bytes = 5u32.to_bytes();
        bytes.push(0);
        assert_eq!(u32::from_bytes(&bytes), Err(DecodeError::TrailingBytes(1)));
    }

    #[test]
    fn hostile_length_prefix_does_not_allocate() {
        let mut enc = Encoder::new();
        enc.put_uvarint(u64::MAX / 2);
        let err = Vec::<u8>::from_bytes(enc.as_bytes()).unwrap_err();
        assert!(matches!(err, DecodeError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_tags_and_utf8() {
        assert!(matches!(
            bool::from_bytes(&[2]),
            Err(DecodeError::InvalidTag { what: "bool", .. })
        ));
        assert!(matches!(
            Option::<u8>::from_bytes(&[9]),
            Err(DecodeError::InvalidTag { what: "Option", .. })
        ));
        assert_eq!(
            String::from_bytes(&[2, 0xC3, 0x28]),
            Err(DecodeError::InvalidUtf8)
        );
    }

    #[test]
    fn out_of_range_narrow_integers() {
        let bytes = 70_000u64.to_bytes();
        assert!(matches!(
            u16::from_bytes(&bytes),
            Err(DecodeError::Invalid(_))
        ));
    }
}
