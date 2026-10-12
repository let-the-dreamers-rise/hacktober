//! Arbitrary-precision signed integers.
//!
//! Chimera integers are `i64` until an operation overflows, at which point the
//! interpreter transparently promotes to [`BigInt`] (and demotes back when a
//! result fits again), so `2 ** 1000` and `factorial(100)` are exact.
//!
//! Representation: sign + magnitude, magnitude as little-endian base-2³² limbs
//! with no trailing zero limbs (zero is the empty vector and never negative).
//!
//! Algorithms: schoolbook addition/subtraction, schoolbook multiplication with
//! Karatsuba above [`KARATSUBA_THRESHOLD`] limbs, and Knuth's Algorithm D
//! (TAOCP vol. 2, §4.3.1) for long division. Division follows Python's
//! *floor* semantics so `-7 // 2 == -4` and `-7 % 2 == 1`.

use std::cmp::Ordering;
use std::fmt;

/// Above this many limbs (per operand), multiplication switches to Karatsuba.
pub const KARATSUBA_THRESHOLD: usize = 40;

/// An arbitrary-precision integer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct BigInt {
    neg: bool,
    mag: Vec<u32>,
}

fn trim(mag: &mut Vec<u32>) {
    while mag.last() == Some(&0) {
        mag.pop();
    }
}

fn cmp_mag(a: &[u32], b: &[u32]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn add_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(long.len() + 1);
    let mut carry = 0u64;
    for (i, &x) in long.iter().enumerate() {
        let s = u64::from(x) + u64::from(short.get(i).copied().unwrap_or(0)) + carry;
        out.push(s as u32);
        carry = s >> 32;
    }
    if carry != 0 {
        out.push(carry as u32);
    }
    out
}

/// `a - b`, requires `a >= b` in magnitude.
fn sub_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for (i, &x) in a.iter().enumerate() {
        let mut d = i64::from(x) - i64::from(b.get(i).copied().unwrap_or(0)) - borrow;
        if d < 0 {
            d += 1 << 32;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(d as u32);
    }
    trim(&mut out);
    out
}

fn mul_schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &y) in b.iter().enumerate() {
            let t = u64::from(x) * u64::from(y) + u64::from(out[i + j]) + carry;
            out[i + j] = t as u32;
            carry = t >> 32;
        }
        let mut k = i + b.len();
        while carry != 0 {
            let t = u64::from(out[k]) + carry;
            out[k] = t as u32;
            carry = t >> 32;
            k += 1;
        }
    }
    trim(&mut out);
    out
}

fn shift_limbs(a: &[u32], n: usize) -> Vec<u32> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0; n];
    out.extend_from_slice(a);
    out
}

fn mul_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.len() < KARATSUBA_THRESHOLD || b.len() < KARATSUBA_THRESHOLD {
        return mul_schoolbook(a, b);
    }
    // Karatsuba: split at m limbs, three recursive products instead of four.
    let m = a.len().max(b.len()) / 2;
    let split = |x: &[u32]| -> (Vec<u32>, Vec<u32>) {
        if x.len() <= m {
            let mut lo = x.to_vec();
            trim(&mut lo);
            (lo, Vec::new())
        } else {
            let mut lo = x[..m].to_vec();
            trim(&mut lo);
            (lo, x[m..].to_vec())
        }
    };
    let (a0, a1) = split(a);
    let (b0, b1) = split(b);
    let z0 = mul_mag(&a0, &b0);
    let z2 = mul_mag(&a1, &b1);
    let z1 = sub_mag(
        &sub_mag(&mul_mag(&add_mag(&a0, &a1), &add_mag(&b0, &b1)), &z0),
        &z2,
    );
    let mut out = add_mag(
        &add_mag(&shift_limbs(&z2, 2 * m), &shift_limbs(&z1, m)),
        &z0,
    );
    trim(&mut out);
    out
}

/// Divides by a single limb, returning (quotient, remainder).
fn divmod_small(a: &[u32], d: u32) -> (Vec<u32>, u32) {
    let mut q = vec![0u32; a.len()];
    let mut rem = 0u64;
    for i in (0..a.len()).rev() {
        let cur = (rem << 32) | u64::from(a[i]);
        q[i] = (cur / u64::from(d)) as u32;
        rem = cur % u64::from(d);
    }
    trim(&mut q);
    (q, rem as u32)
}

fn shl_bits(a: &[u32], s: u32) -> Vec<u32> {
    if s == 0 {
        return a.to_vec();
    }
    let mut out = Vec::with_capacity(a.len() + 1);
    let mut carry = 0u32;
    for &x in a {
        out.push((x << s) | carry);
        carry = x >> (32 - s);
    }
    out.push(carry);
    out
}

/// Long division of magnitudes (Knuth Algorithm D). `v` must be non-zero.
fn divmod_mag(u: &[u32], v: &[u32]) -> (Vec<u32>, Vec<u32>) {
    if cmp_mag(u, v) == Ordering::Less {
        return (Vec::new(), u.to_vec());
    }
    if v.len() == 1 {
        let (q, r) = divmod_small(u, v[0]);
        let r = if r == 0 { Vec::new() } else { vec![r] };
        return (q, r);
    }
    let n = v.len();
    let m = u.len() - n;
    let s = v[n - 1].leading_zeros();
    let vn = {
        let mut x = shl_bits(v, s);
        x.truncate(n);
        x
    };
    let mut un = shl_bits(u, s);
    if un.len() == u.len() {
        un.push(0);
    }
    let b: u64 = 1 << 32;
    let mut q = vec![0u32; m + 1];
    for j in (0..=m).rev() {
        let num = (u64::from(un[j + n]) << 32) | u64::from(un[j + n - 1]);
        let mut qhat = num / u64::from(vn[n - 1]);
        let mut rhat = num % u64::from(vn[n - 1]);
        while qhat >= b || qhat * u64::from(vn[n - 2]) > ((rhat << 32) | u64::from(un[j + n - 2])) {
            qhat -= 1;
            rhat += u64::from(vn[n - 1]);
            if rhat >= b {
                break;
            }
        }
        // Multiply and subtract qhat * vn from un[j..=j+n].
        let mut borrow = 0i64;
        let mut carry = 0u64;
        for i in 0..n {
            let p = qhat * u64::from(vn[i]) + carry;
            carry = p >> 32;
            let t = i64::from(un[i + j]) - borrow - (p & 0xFFFF_FFFF) as i64;
            un[i + j] = t as u32;
            borrow = i64::from(t < 0);
        }
        let t = i64::from(un[j + n]) - borrow - carry as i64;
        un[j + n] = t as u32;
        if t < 0 {
            // qhat was one too large: add the divisor back.
            qhat -= 1;
            let mut c = 0u64;
            for i in 0..n {
                let sum = u64::from(un[i + j]) + u64::from(vn[i]) + c;
                un[i + j] = sum as u32;
                c = sum >> 32;
            }
            un[j + n] = un[j + n].wrapping_add(c as u32);
        }
        q[j] = qhat as u32;
    }
    trim(&mut q);
    // Un-normalise the remainder.
    let mut r = vec![0u32; n];
    for i in 0..n {
        r[i] = if s == 0 {
            un[i]
        } else {
            (un[i] >> s) | (un[i + 1] << (32 - s))
        };
    }
    trim(&mut r);
    (q, r)
}

impl BigInt {
    /// Zero.
    pub fn zero() -> Self {
        BigInt::default()
    }

    fn from_parts(neg: bool, mut mag: Vec<u32>) -> Self {
        trim(&mut mag);
        let neg = neg && !mag.is_empty();
        BigInt { neg, mag }
    }

    /// Converts from `u64`.
    pub fn from_u64(v: u64) -> Self {
        BigInt::from_parts(false, vec![v as u32, (v >> 32) as u32])
    }

    /// Converts from `i64`.
    pub fn from_i64(v: i64) -> Self {
        let mut b = BigInt::from_u64(v.unsigned_abs());
        b.neg = v < 0;
        b
    }

    /// Converts a finite float, truncating toward zero.
    pub fn from_f64(f: f64) -> Option<Self> {
        if !f.is_finite() {
            return None;
        }
        let f = f.trunc();
        if f.abs() < 9.0e18 {
            return Some(BigInt::from_i64(f as i64));
        }
        let bits = f.abs().to_bits();
        let exp = ((bits >> 52) & 0x7FF) as i64 - 1075;
        let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
        let mut b = BigInt::from_u64(mantissa);
        b = b.shl(exp as u32);
        b.neg = f < 0.0;
        Some(b)
    }

    fn shl(&self, bits: u32) -> BigInt {
        let limbs = (bits / 32) as usize;
        let mag = shift_limbs(&shl_bits(&self.mag, bits % 32), limbs);
        BigInt::from_parts(self.neg, mag)
    }

    /// The value as `i64`, if it fits.
    pub fn to_i64(&self) -> Option<i64> {
        if self.mag.len() > 2 {
            return None;
        }
        let lo = u64::from(self.mag.first().copied().unwrap_or(0));
        let hi = u64::from(self.mag.get(1).copied().unwrap_or(0));
        let m = (hi << 32) | lo;
        if self.neg {
            if m <= i64::MAX as u64 + 1 {
                Some((m as i64).wrapping_neg())
            } else {
                None
            }
        } else {
            i64::try_from(m).ok()
        }
    }

    /// Nearest float (may be infinite for huge values).
    pub fn to_f64(&self) -> f64 {
        let mut f = 0.0f64;
        for &limb in self.mag.iter().rev() {
            f = f * 4_294_967_296.0 + f64::from(limb);
        }
        if self.neg { -f } else { f }
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    /// Whether the value is negative.
    pub fn is_negative(&self) -> bool {
        self.neg
    }

    /// Parses an optionally signed decimal string.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let mut b = BigInt::parse_radix(digits, 10)?;
        b.neg = neg && !b.mag.is_empty();
        Some(b)
    }

    /// Parses unsigned digits in the given radix (2..=36), ignoring `_`.
    pub fn parse_radix(digits: &str, radix: u32) -> Option<Self> {
        if digits.is_empty() {
            return None;
        }
        let mut mag: Vec<u32> = Vec::new();
        for c in digits.chars().filter(|&c| c != '_') {
            let d = c.to_digit(radix)?;
            // mag = mag * radix + d
            let mut carry = u64::from(d);
            for limb in &mut mag {
                let t = u64::from(*limb) * u64::from(radix) + carry;
                *limb = t as u32;
                carry = t >> 32;
            }
            if carry != 0 {
                mag.push(carry as u32);
            }
        }
        Some(BigInt::from_parts(false, mag))
    }

    /// `-self`.
    pub fn neg(&self) -> BigInt {
        BigInt::from_parts(!self.neg, self.mag.clone())
    }

    /// `|self|`.
    pub fn abs(&self) -> BigInt {
        BigInt::from_parts(false, self.mag.clone())
    }

    /// `self + other`.
    pub fn add(&self, other: &BigInt) -> BigInt {
        if self.neg == other.neg {
            return BigInt::from_parts(self.neg, add_mag(&self.mag, &other.mag));
        }
        match cmp_mag(&self.mag, &other.mag) {
            Ordering::Equal => BigInt::zero(),
            Ordering::Greater => BigInt::from_parts(self.neg, sub_mag(&self.mag, &other.mag)),
            Ordering::Less => BigInt::from_parts(other.neg, sub_mag(&other.mag, &self.mag)),
        }
    }

    /// `self - other`.
    pub fn sub(&self, other: &BigInt) -> BigInt {
        self.add(&other.neg())
    }

    /// `self * other`.
    pub fn mul(&self, other: &BigInt) -> BigInt {
        BigInt::from_parts(self.neg != other.neg, mul_mag(&self.mag, &other.mag))
    }

    /// Truncating division: quotient rounds toward zero, remainder has the
    /// dividend's sign. `None` on division by zero.
    pub fn divmod_trunc(&self, other: &BigInt) -> Option<(BigInt, BigInt)> {
        if other.is_zero() {
            return None;
        }
        let (q, r) = divmod_mag(&self.mag, &other.mag);
        Some((
            BigInt::from_parts(self.neg != other.neg, q),
            BigInt::from_parts(self.neg, r),
        ))
    }

    /// Floor division (Python semantics): quotient rounds toward −∞ and the
    /// remainder has the divisor's sign. `None` on division by zero.
    pub fn divmod_floor(&self, other: &BigInt) -> Option<(BigInt, BigInt)> {
        let (q, r) = self.divmod_trunc(other)?;
        if !r.is_zero() && (r.neg != other.neg) {
            Some((q.sub(&BigInt::from_i64(1)), r.add(other)))
        } else {
            Some((q, r))
        }
    }

    /// `self ** exp`.
    pub fn pow(&self, mut exp: u64) -> BigInt {
        let mut base = self.clone();
        let mut acc = BigInt::from_i64(1);
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.mul(&base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base);
            }
        }
        acc
    }

    /// Number of decimal digits in `|self|` (1 for zero).
    pub fn decimal_len(&self) -> usize {
        self.abs().to_string().len()
    }

    /// Formats in the given radix (2..=36), lowercase digits.
    pub fn to_string_radix(&self, radix: u32) -> String {
        if self.is_zero() {
            return "0".to_owned();
        }
        // Peel off chunks of digits using the largest power of radix < 2^32.
        let mut chunk_pow = radix;
        let mut chunk_digits = 1;
        while u64::from(chunk_pow) * u64::from(radix) < (1u64 << 32) {
            chunk_pow *= radix;
            chunk_digits += 1;
        }
        let mut chunks = Vec::new();
        let mut mag = self.mag.clone();
        while !mag.is_empty() {
            let (q, r) = divmod_small(&mag, chunk_pow);
            chunks.push(r);
            mag = q;
        }
        let digit = |d: u32| std::char::from_digit(d, radix).unwrap_or('?');
        let mut out = String::new();
        if self.neg {
            out.push('-');
        }
        for (i, &chunk) in chunks.iter().rev().enumerate() {
            let mut s = String::new();
            let mut c = chunk;
            loop {
                s.push(digit(c % radix));
                c /= radix;
                if c == 0 {
                    break;
                }
            }
            if i > 0 {
                while s.len() < chunk_digits {
                    s.push('0');
                }
            }
            out.extend(s.chars().rev());
        }
        out
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(&self.mag, &other.mag),
            (true, true) => cmp_mag(&other.mag, &self.mag),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_radix(10))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chimera_core::Rng;

    fn big(s: &str) -> BigInt {
        BigInt::parse(s).unwrap()
    }

    #[test]
    fn parse_and_display_roundtrip() {
        for s in [
            "0",
            "1",
            "-1",
            "4294967295",
            "4294967296",
            "-18446744073709551616",
            "123456789012345678901234567890123456789",
        ] {
            assert_eq!(big(s).to_string(), s);
        }
        assert_eq!(big("-0").to_string(), "0");
        assert_eq!(BigInt::parse_radix("ff", 16).unwrap().to_string(), "255");
        assert_eq!(big("255").to_string_radix(2), "11111111");
        assert!(BigInt::parse("12a").is_none());
    }

    #[test]
    fn i64_conversions() {
        for v in [0, 1, -1, i64::MAX, i64::MIN, 1 << 40] {
            assert_eq!(BigInt::from_i64(v).to_i64(), Some(v));
        }
        assert_eq!(big("9223372036854775808").to_i64(), None);
        assert_eq!(big("-9223372036854775808").to_i64(), Some(i64::MIN));
        assert_eq!(big("1000000").to_f64(), 1e6);
        assert_eq!(
            BigInt::from_f64(1e20).unwrap().to_string(),
            "100000000000000000000"
        );
        assert_eq!(BigInt::from_f64(-3.9).unwrap().to_string(), "-3");
    }

    #[test]
    fn arithmetic_matches_i128() {
        let mut rng = Rng::new(1234);
        for _ in 0..2_000 {
            let a = rng.next_u64() as i64 as i128 * rng.range_i64(-1000, 1000) as i128;
            let b = rng.next_u64() as i64 as i128 >> rng.below(60);
            let (ba, bb) = (big(&a.to_string()), big(&b.to_string()));
            assert_eq!(ba.add(&bb).to_string(), (a + b).to_string());
            assert_eq!(ba.sub(&bb).to_string(), (a - b).to_string());
            if let Some(p) = a.checked_mul(b) {
                assert_eq!(ba.mul(&bb).to_string(), p.to_string());
            }
            if b != 0 {
                let (q, r) = ba.divmod_floor(&bb).unwrap();
                // Python floor division, computed with i128.
                let (mut eq, mut er) = (a / b, a % b);
                if er != 0 && ((er < 0) != (b < 0)) {
                    eq -= 1;
                    er += b;
                }
                assert_eq!(q.to_string(), eq.to_string());
                assert_eq!(r.to_string(), er.to_string());
            }
        }
    }

    #[test]
    fn multiplication_and_division_of_huge_numbers() {
        let a = big("2").pow(4000).sub(&big("1"));
        let b = big("3").pow(1500).add(&big("7"));
        let p = a.mul(&b);
        // Karatsuba agrees with schoolbook.
        assert_eq!(BigInt::from_parts(false, mul_schoolbook(&a.mag, &b.mag)), p);
        let (q, r) = p.add(&big("5")).divmod_floor(&b).unwrap();
        assert_eq!(q, a);
        assert_eq!(r, big("5"));
        let (q2, r2) = a.divmod_trunc(&b).unwrap();
        assert_eq!(q2.mul(&b).add(&r2), a);
        assert!(r2 < b);
    }

    #[test]
    fn floor_semantics_on_negatives() {
        let (q, r) = big("-7").divmod_floor(&big("2")).unwrap();
        assert_eq!((q.to_string(), r.to_string()), ("-4".into(), "1".into()));
        let (q, r) = big("7").divmod_floor(&big("-2")).unwrap();
        assert_eq!((q.to_string(), r.to_string()), ("-4".into(), "-1".into()));
        assert!(big("1").divmod_floor(&BigInt::zero()).is_none());
    }

    #[test]
    fn known_big_values() {
        assert_eq!(
            big("2").pow(100).to_string(),
            "1267650600228229401496703205376"
        );
        let mut fact = big("1");
        for i in 1..=30 {
            fact = fact.mul(&BigInt::from_i64(i));
        }
        assert_eq!(fact.to_string(), "265252859812191058636308480000000");
        assert_eq!(big("2").pow(1000).decimal_len(), 302);
    }

    #[test]
    fn ordering() {
        let mut xs = vec![
            big("5"),
            big("-100"),
            big("99999999999999999999"),
            big("0"),
            big("-3"),
        ];
        xs.sort();
        let s: Vec<String> = xs.iter().map(ToString::to_string).collect();
        assert_eq!(s, ["-100", "-3", "0", "5", "99999999999999999999"]);
    }
}
