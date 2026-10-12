//! Deterministic pseudo-random numbers.
//!
//! [`Rng`] is xoshiro256\*\* seeded through SplitMix64. It is *not*
//! cryptographically secure; it is fast, statistically solid and, most
//! importantly, reproducible: the same seed yields the same stream on every
//! platform. The Raft simulator replays failures from a single `u64` seed and
//! the renderer derives one independent stream per pixel, so images are
//! identical no matter how many threads render them.

/// Advances a SplitMix64 state and returns the next output.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mix64(*state)
}

/// A stateless 64-bit finaliser (the SplitMix64 output function).
///
/// Useful for hashing small integers such as pixel coordinates into seeds.
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seedable xoshiro256\*\* generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Creates a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Rng { s }
    }

    /// Creates a generator for one of many independent streams derived from a
    /// shared seed, e.g. one stream per pixel or per simulated node.
    pub fn from_stream(seed: u64, stream: u64) -> Self {
        Rng::new(seed ^ mix64(stream.wrapping_add(0x632B_E59B_D9B4_E019)))
    }

    /// Splits off a new generator whose stream is independent of this one.
    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Returns the next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f32` in `[0, 1)` with 24 bits of precision.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform integer in `[0, n)` without modulo bias (Lemire's method).
    ///
    /// # Panics
    /// Panics if `n == 0`.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below called with n == 0");
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Uniform index in `[0, len)`.
    ///
    /// # Panics
    /// Panics if `len == 0`.
    pub fn index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    /// Uniform integer in `[lo, hi)`.
    ///
    /// # Panics
    /// Panics if `lo >= hi`.
    pub fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        assert!(lo < hi, "Rng::range_i64 called with an empty range");
        let span = hi.wrapping_sub(lo) as u64;
        lo.wrapping_add(self.below(span) as i64)
    }

    /// Uniform float in `[lo, hi)`.
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Returns `true` with probability `p` (clamped to `[0, 1]`).
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// A normally distributed sample (Box–Muller transform).
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        // 1 - u keeps the argument of ln strictly positive.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let mag = (-2.0 * u1.ln()).sqrt();
        mean + std_dev * mag * (std::f64::consts::TAU * u2).cos()
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.index(i + 1);
            items.swap(i, j);
        }
    }

    /// Picks a uniformly random element, or `None` if `items` is empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.index(items.len())])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let same = (0..64).filter(|_| a.next_u64() == b.next_u64()).count();
        assert_eq!(same, 0);
    }

    #[test]
    fn streams_are_independent() {
        let mut a = Rng::from_stream(7, 0);
        let mut b = Rng::from_stream(7, 1);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn floats_are_in_unit_interval() {
        let mut rng = Rng::new(3);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            let y = rng.next_f32();
            assert!((0.0..1.0).contains(&y));
        }
    }

    #[test]
    fn below_is_in_range_and_roughly_uniform() {
        let mut rng = Rng::new(99);
        let mut buckets = [0u32; 6];
        for _ in 0..60_000 {
            let x = rng.below(6);
            buckets[x as usize] += 1;
        }
        for count in buckets {
            assert!((9_000..11_000).contains(&count), "bucket count {count}");
        }
    }

    #[test]
    fn range_i64_handles_negative_and_extreme_bounds() {
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            let x = rng.range_i64(-10, 10);
            assert!((-10..10).contains(&x));
        }
        let x = rng.range_i64(i64::MIN, i64::MAX);
        assert!(x < i64::MAX);
    }

    #[test]
    fn normal_has_expected_moments() {
        let mut rng = Rng::new(11);
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal(5.0, 2.0)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 5.0).abs() < 0.05, "mean {mean}");
        assert!((var.sqrt() - 2.0).abs() < 0.05, "std {}", var.sqrt());
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(8);
        let mut xs: Vec<u32> = (0..100).collect();
        rng.shuffle(&mut xs);
        assert_ne!(xs, (0..100).collect::<Vec<_>>());
        xs.sort_unstable();
        assert_eq!(xs, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn choose_handles_empty() {
        let mut rng = Rng::new(0);
        let empty: [u8; 0] = [];
        assert_eq!(rng.choose(&empty), None);
        assert_eq!(rng.choose(&[9]), Some(&9));
    }
}
