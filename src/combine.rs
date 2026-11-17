//! This module provides a function to combine CRCs of two sequences of bytes.
//!
//! It is based on the work of Mark Adler and is designed to be used with
//! different CRC algorithms.

use crate::{CrcAlgorithm, CrcParams};
use core::num::NonZeroU64;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use crate::{
    feature_detection::{get_arch_ops, ArchOpsInstance},
    structs::{CrcState, Width16, Width32, Width64},
    traits::{ArchOps, EnhancedCrcWidth},
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use core::marker::PhantomData;

#[inline(always)]
pub fn checksums(crc1: u64, crc2: u64, len2: u64, params: &CrcParams) -> u64 {
    if params.refin != params.refout {
        panic!("Unsupported CRC configuration");
    }
    let crc1 = crc1 ^ (params.init_algorithm ^ params.xorout);
    match NonZeroU64::new(len2) {
        Some(len2) => shift(crc1, len2, params, Powers::of(params)) ^ crc2,
        None => crc1 ^ crc2,
    }
}

#[inline(always)]
fn shift(crc: u64, len: NonZeroU64, params: &CrcParams, powers: Option<&Powers>) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let has_constant_term = params.poly & 1 == 1;
        if has_constant_term && has_clmul() {
            match params.width {
                64 => return unsafe { shift_clmul::<Width64>(crc, len, params, powers) },
                32 => return unsafe { shift_clmul::<Width32>(crc, len, params, powers) },
                16 => return unsafe { shift_clmul::<Width16>(crc, len, params, powers) },
                _ => {}
            }
        }
    }
    shift_software(crc, len, params, powers)
}

fn shift_software(crc: u64, len: NonZeroU64, params: &CrcParams, powers: Option<&Powers>) -> u64 {
    let modulus = Modulus::new(params.width, params.poly, params.refin);
    shift_with(&modulus, crc & modulus.mask(), len, params, powers)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn has_clmul() -> bool {
    cfg!(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_feature = "sse4.1", target_feature = "pclmulqdq"),
    )) || !matches!(get_arch_ops(), ArchOpsInstance::SoftwareFallback)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn shift_clmul<W: EnhancedCrcWidth>(
    crc: u64,
    len: NonZeroU64,
    params: &CrcParams,
    powers: Option<&Powers>,
) -> u64
where
    W::Value: Into<u64>,
{
    let ops = crate::arch::aarch64::aes::Aarch64AesOps;
    shift_clmul_with::<_, W>(&ops, crc, len, params, powers)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "sse4.1,pclmulqdq")]
unsafe fn shift_clmul<W: EnhancedCrcWidth>(
    crc: u64,
    len: NonZeroU64,
    params: &CrcParams,
    powers: Option<&Powers>,
) -> u64
where
    W::Value: Into<u64>,
{
    let ops = crate::arch::x86::sse::X86SsePclmulqdqOps;
    shift_clmul_with::<_, W>(&ops, crc, len, params, powers)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn shift_clmul_with<T: ArchOps, W: EnhancedCrcWidth>(
    ops: &T,
    crc: u64,
    len: NonZeroU64,
    params: &CrcParams,
    powers: Option<&Powers>,
) -> u64
where
    T::Vector: Copy,
    W::Value: Into<u64>,
{
    shift_with(&Clmul::<T, W>::new(ops, params), crc, len, params, powers)
}

#[inline(always)]
fn shift_with<M: Multiply>(
    multiplier: &M,
    crc: u64,
    len: NonZeroU64,
    params: &CrcParams,
    powers: Option<&Powers>,
) -> u64 {
    let factor = match powers {
        Some(powers) => powers.factor(multiplier, len),
        None => Modulus::new(params.width, params.poly, params.refin).factor(multiplier, len),
    };
    multiplier.multiply(crc, factor)
}

trait Multiply {
    fn multiply(&self, a: u64, b: u64) -> u64;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
struct Clmul<'a, T, W> {
    ops: &'a T,
    reflected: bool,
    mu: u64,
    poly: u64,
    width: PhantomData<W>,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
impl<'a, T, W> Clmul<'a, T, W> {
    #[inline(always)]
    unsafe fn new(ops: &'a T, params: &CrcParams) -> Self {
        Self {
            ops,
            reflected: params.refin,
            mu: params.get_key(7),
            poly: params.get_key(8),
            width: PhantomData,
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
impl<T: ArchOps, W: EnhancedCrcWidth> Multiply for Clmul<'_, T, W>
where
    T::Vector: Copy,
    W::Value: Into<u64>,
{
    #[inline(always)]
    fn multiply(&self, a: u64, b: u64) -> u64 {
        unsafe {
            let product = self.ops.carryless_mul_00(
                self.ops
                    .create_vector_from_u64(a & (u64::MAX >> (64 - W::WIDTH)), false),
                self.ops.create_vector_from_u64(b << (64 - W::WIDTH), false),
            );
            let state = CrcState {
                value: product,
                reflected: self.reflected,
            };
            W::barrett_reduction(&state, self.poly, self.mu, self.ops).into()
        }
    }
}

struct Modulus {
    width: u8,
    poly: u64,
    reflected: bool,
}

impl Modulus {
    const fn new(width: u8, poly: u64, reflected: bool) -> Self {
        assert!(width >= 1 && width <= 64, "Unsupported CRC width");
        let poly = if reflected {
            poly.reverse_bits() >> (64 - width)
        } else {
            poly & (u64::MAX >> (64 - width))
        };
        Self {
            width,
            poly,
            reflected,
        }
    }

    const fn mask(&self) -> u64 {
        u64::MAX >> (64 - self.width)
    }

    const fn times_x(&self, value: u64) -> u64 {
        if self.reflected {
            (value >> 1) ^ if value & 1 == 1 { self.poly } else { 0 }
        } else {
            let carry = (value >> (self.width - 1)) & 1 == 1;
            ((value << 1) & self.mask()) ^ if carry { self.poly } else { 0 }
        }
    }

    const fn multiply_bitwise(&self, a: u64, b: u64) -> u64 {
        let mut product = 0;
        let mut bit = 0;
        while bit < self.width {
            product = if self.reflected {
                self.times_x(product ^ if (b >> bit) & 1 == 1 { a } else { 0 })
            } else {
                let set = (b >> (self.width - 1 - bit)) & 1 == 1;
                self.times_x(product) ^ if set { a } else { 0 }
            };
            bit += 1;
        }
        product
    }

    const fn byte_shift(&self) -> u64 {
        let (mut power, mut steps) = if self.reflected {
            (1 << (self.width - 1), 7)
        } else {
            (1, 8)
        };
        while steps > 0 {
            power = self.times_x(power);
            steps -= 1;
        }
        power
    }

    #[inline(always)]
    fn factor<M: Multiply>(&self, multiplier: &M, len: NonZeroU64) -> u64 {
        let mut power = self.byte_shift();
        for _ in 0..len.trailing_zeros() {
            power = multiplier.multiply(power, power);
        }
        let mut factor = power;
        let mut rest = len.get() >> len.trailing_zeros() >> 1;
        while rest != 0 {
            power = multiplier.multiply(power, power);
            if rest & 1 == 1 {
                factor = multiplier.multiply(factor, power);
            }
            rest >>= 1;
        }
        factor
    }
}

impl Multiply for Modulus {
    fn multiply(&self, a: u64, b: u64) -> u64 {
        self.multiply_bitwise(a, b)
    }
}

struct Powers {
    width: u8,
    poly: u64,
    reflected: bool,
    table: [[u64; 16]; 16],
}

impl Powers {
    const fn new(width: u8, poly: u64, reflected: bool) -> Self {
        let modulus = Modulus::new(width, poly, reflected);
        let mut table = [[0; 16]; 16];
        let mut power = modulus.byte_shift();
        let mut digit = 0;
        while digit < 16 {
            table[digit][1] = power;
            let mut count = 2;
            while count < 16 {
                table[digit][count] = modulus.multiply_bitwise(table[digit][count - 1], power);
                count += 1;
            }
            power = modulus.multiply_bitwise(table[digit][15], power);
            digit += 1;
        }
        Self {
            width,
            poly,
            reflected,
            table,
        }
    }

    #[inline]
    #[allow(deprecated)]
    fn of(params: &CrcParams) -> Option<&'static Self> {
        use CrcAlgorithm::*;
        macro_rules! powers {
            ($width:literal, $poly:literal, $reflected:literal) => {{
                static POWERS: Powers = Powers::new($width, $poly, $reflected);
                &POWERS
            }};
        }
        let powers = match params.algorithm {
            Crc16DectR | Crc16DectX => powers!(16, 0x0589, false),
            Crc16Nrsc5 => powers!(16, 0x080b, true),
            Crc16Genibus | Crc16Gsm | Crc16Ibm3740 | Crc16SpiFujitsu | Crc16Xmodem => {
                powers!(16, 0x1021, false)
            }
            Crc16IbmSdlc | Crc16IsoIec144433A | Crc16Kermit | Crc16Mcrf4xx | Crc16Riello
            | Crc16Tms37157 => powers!(16, 0x1021, true),
            Crc16Profibus => powers!(16, 0x1dcf, false),
            Crc16En13757 => powers!(16, 0x3d65, false),
            Crc16Dnp => powers!(16, 0x3d65, true),
            Crc16M17 | Crc16OpensafetyA => powers!(16, 0x5935, false),
            Crc16Lj1200 => powers!(16, 0x6f63, false),
            Crc16OpensafetyB => powers!(16, 0x755b, false),
            Crc16Cms | Crc16Dds110 | Crc16Umts => powers!(16, 0x8005, false),
            Crc16Arc | Crc16MaximDow | Crc16Modbus | Crc16Usb => powers!(16, 0x8005, true),
            Crc16T10Dif => powers!(16, 0x8bb7, false),
            Crc16Teledisk => powers!(16, 0xa097, false),
            Crc16Cdma2000 => powers!(16, 0xc867, false),
            Crc32Iscsi => powers!(32, 0x1edc6f41, true),
            Crc32Bzip2 | Crc32Cksum | Crc32Mpeg2 => powers!(32, 0x04c11db7, false),
            Crc32IsoHdlc | Crc32Jamcrc => powers!(32, 0x04c11db7, true),
            Crc32Mef => powers!(32, 0x741b8cd7, true),
            Crc32CdRomEdc => powers!(32, 0x8001801b, true),
            Crc32Aixm => powers!(32, 0x814141ab, false),
            Crc32Base91D => powers!(32, 0xa833982b, true),
            Crc32Xfer => powers!(32, 0x000000af, false),
            Crc32Autosar => powers!(32, 0xf4acfb13, true),
            Crc64GoIso => powers!(64, 0x000000000000001b, true),
            Crc64Ms => powers!(64, 0x259c84cba6426349, true),
            Crc64Ecma182 | Crc64We => powers!(64, 0x42f0e1eba9ea3693, false),
            Crc64Xz => powers!(64, 0x42f0e1eba9ea3693, true),
            Crc64Redis => powers!(64, 0xad93d23594c935a9, true),
            Crc64Nvme => powers!(64, 0xad93d23594c93659, true),
            CrcCustom | Crc32Custom | Crc64Custom => return None,
        };
        let matches = powers.width == params.width
            && powers.poly == params.poly
            && powers.reflected == params.refin;
        matches.then_some(powers)
    }

    #[inline(always)]
    fn factor<M: Multiply>(&self, multiplier: &M, len: NonZeroU64) -> u64 {
        let mut digits = len.get();
        let mut factor = self.take(&mut digits);
        while digits != 0 {
            factor = multiplier.multiply(factor, self.take(&mut digits));
        }
        factor
    }

    #[inline(always)]
    fn take(&self, digits: &mut u64) -> u64 {
        let shift = digits.trailing_zeros() & 60;
        let power = self.table[(shift / 4) as usize][((*digits >> shift) & 15) as usize];
        *digits &= !(15 << shift);
        power
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum_with_params;
    use crate::test::consts::TEST_ALL_CONFIGS;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn custom(width: u8, poly: u64, reflected: bool) -> CrcParams {
        let [init, xorout] = [0x0f1e2d3c4b5a6978, 0x8796a5b4c3d2e1f0u64].map(|x| x >> (64 - width));
        let params = CrcParams::new("custom", width, poly, init, reflected, xorout, 0);
        assert!(Powers::of(&params).is_none());
        params
    }

    fn all_params() -> Vec<CrcParams> {
        let custom = [(16, 0x1337), (32, 0x1234567b), (64, 0x123456789abcdef1)]
            .into_iter()
            .flat_map(|(width, poly)| {
                [false, true].map(|reflected| custom(width, poly, reflected))
            });
        TEST_ALL_CONFIGS
            .iter()
            .map(|config| *config.get_params())
            .chain(custom)
            .collect()
    }

    fn shifts(params: &CrcParams, crc: u64, len: NonZeroU64) -> [u64; 4] {
        let powers = Powers::of(params);
        [
            shift(crc, len, params, powers),
            shift(crc, len, params, None),
            shift_software(crc, len, params, powers),
            shift_software(crc, len, params, None),
        ]
    }

    #[test]
    fn tables_cover_exactly_the_predefined_polynomials() {
        for config in TEST_ALL_CONFIGS {
            let params = *config.get_params();
            assert!(Powers::of(&params).is_some(), "{}", params.name);
            for other in [
                CrcParams {
                    width: params.width / 2,
                    ..params
                },
                CrcParams {
                    poly: params.poly ^ 2,
                    ..params
                },
                CrcParams {
                    refin: !params.refin,
                    ..params
                },
            ] {
                assert!(Powers::of(&other).is_none(), "{}", params.name);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn combine_matches_checksum_of_concatenation() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut data = vec![0u8; 70_000];
        rng.fill(&mut data[..]);
        for params in all_params() {
            for round in 0..8 {
                let end = rng.random_range(0..=data.len() >> (2 * round));
                let expected = checksum_with_params(params, &data[..end]);
                for split in [0, end, rng.random_range(0..=end)] {
                    let (a, b) = data[..end].split_at(split);
                    let crc_a = checksum_with_params(params, a);
                    let crc_b = checksum_with_params(params, b);
                    let len_b = b.len() as u64;
                    assert_eq!(checksums(crc_a, crc_b, len_b, &params), expected);
                    if let Some(len_b) = NonZeroU64::new(len_b) {
                        let crc_a = crc_a ^ params.init_algorithm ^ params.xorout;
                        let expected = [expected ^ crc_b; 4];
                        assert_eq!(shifts(&params, crc_a, len_b), expected, "{}", params.name);
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shifts_agree_and_compose_at_any_length() {
        let mut rng = StdRng::seed_from_u64(2);
        let without_constant_term = [16, 32, 64]
            .into_iter()
            .flat_map(|width| [false, true].map(|reflected| custom(width, 0x1336, reflected)));
        for params in all_params().into_iter().chain(without_constant_term) {
            let powers = Powers::of(&params);
            let mut lens = vec![1, u64::MAX, 1 << 63, (1 << 61) + 1];
            lens.extend((0..64).map(|_| rng.random::<u64>() >> rng.random_range(0..64)));
            lens.extend((0..16).map(|_| 1 << rng.random_range(0..64)));
            for len in lens.into_iter().filter_map(NonZeroU64::new) {
                let crc = rng.random();
                let [shifted, others @ ..] = shifts(&params, crc, len);
                assert_eq!(others, [shifted; 3], "{} {len}", params.name);

                let low = NonZeroU64::new(rng.random_range(1..=len.get())).unwrap();
                let stepwise = match NonZeroU64::new(len.get() - low.get()) {
                    Some(high) => shift(shift(crc, low, &params, powers), high, &params, powers),
                    None => shifted,
                };
                assert_eq!(stepwise, shifted, "{} {len} {low}", params.name);
            }
        }
    }
}
