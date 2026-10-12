//! Checksums and non-cryptographic hashes.
//!
//! * [`crc32`] / [`Crc32`] – CRC-32 (IEEE 802.3, the zlib/PNG polynomial),
//!   matching Python's `zlib.crc32`.
//! * [`adler32`] – the zlib stream checksum.
//! * [`fnv1a64`] – a stable 64-bit hash. Unlike `std`'s `DefaultHasher` its
//!   output never changes between Rust releases, so it is safe to persist.

const fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = make_crc_table();

/// An incremental CRC-32 hasher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    /// Starts a new checksum.
    pub const fn new() -> Self {
        Crc32 { state: 0xFFFF_FFFF }
    }

    /// Feeds more bytes into the checksum.
    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.state;
        for &byte in data {
            c = CRC_TABLE[((c ^ u32::from(byte)) & 0xFF) as usize] ^ (c >> 8);
        }
        self.state = c;
    }

    /// Returns the checksum of everything fed so far.
    pub const fn finish(&self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Crc32::new()
    }
}

/// CRC-32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finish()
}

/// Adler-32 of a byte slice (RFC 1950).
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    // 5552 is the largest n such that 255n(n+1)/2 + (n+1)(MOD-1) fits in u32,
    // so the modulo can be deferred across whole chunks.
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// 64-bit FNV-1a hash.
pub const fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut i = 0;
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        i += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_incremental_matches_one_shot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut hasher = Crc32::new();
        for chunk in data.chunks(5) {
            hasher.update(chunk);
        }
        assert_eq!(hasher.finish(), crc32(data));
        assert_eq!(crc32(data), 0x414F_A339);
    }

    #[test]
    fn adler32_known_values() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
        // Long input exercises the deferred modulo.
        let big = vec![0xFFu8; 100_000];
        let (mut a, mut b) = (1u64, 0u64);
        for &x in &big {
            a = (a + u64::from(x)) % 65_521;
            b = (b + a) % 65_521;
        }
        assert_eq!(adler32(&big), ((b << 16) | a) as u32);
    }

    #[test]
    fn fnv1a64_known_values() {
        assert_eq!(fnv1a64(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_F739_67E8);
    }
}
