use super::*;

impl Vaes {
	delegate!(
		#[target_feature(enable = "vaes")]
		{
			fn _mm256_aesenc_epi128(a: __m256i, round_key: __m256i) -> __m256i;
			fn _mm256_aesenclast_epi128(a: __m256i, round_key: __m256i) -> __m256i;
			fn _mm256_aesdec_epi128(a: __m256i, round_key: __m256i) -> __m256i;
			fn _mm256_aesdeclast_epi128(a: __m256i, round_key: __m256i) -> __m256i;
		}
	);

	// The 512-bit VAES ops require both `vaes` and `avx512f`, but no single
	// pulp token proves both, so the AVX-512F token is taken as an extra
	// proof parameter.
	/// Requires `vaes` (`self`) and `avx512f` (proof parameter).
	#[inline(always)]
	pub fn _mm512_aesenc_epi128(
		self,
		_avx512f: Avx512f,
		a: __m512i,
		round_key: __m512i,
	) -> __m512i {
		#[target_feature(enable = "vaes,avx512f")]
		fn __safety_check__(a: __m512i, round_key: __m512i) -> __m512i {
			arch::_mm512_aesenc_epi128(a, round_key)
		}
		unsafe { arch::_mm512_aesenc_epi128(a, round_key) }
	}

	/// Requires `vaes` (`self`) and `avx512f` (proof parameter).
	#[inline(always)]
	pub fn _mm512_aesenclast_epi128(
		self,
		_avx512f: Avx512f,
		a: __m512i,
		round_key: __m512i,
	) -> __m512i {
		#[target_feature(enable = "vaes,avx512f")]
		fn __safety_check__(a: __m512i, round_key: __m512i) -> __m512i {
			arch::_mm512_aesenclast_epi128(a, round_key)
		}
		unsafe { arch::_mm512_aesenclast_epi128(a, round_key) }
	}

	/// Requires `vaes` (`self`) and `avx512f` (proof parameter).
	#[inline(always)]
	pub fn _mm512_aesdec_epi128(
		self,
		_avx512f: Avx512f,
		a: __m512i,
		round_key: __m512i,
	) -> __m512i {
		#[target_feature(enable = "vaes,avx512f")]
		fn __safety_check__(a: __m512i, round_key: __m512i) -> __m512i {
			arch::_mm512_aesdec_epi128(a, round_key)
		}
		unsafe { arch::_mm512_aesdec_epi128(a, round_key) }
	}

	/// Requires `vaes` (`self`) and `avx512f` (proof parameter).
	#[inline(always)]
	pub fn _mm512_aesdeclast_epi128(
		self,
		_avx512f: Avx512f,
		a: __m512i,
		round_key: __m512i,
	) -> __m512i {
		#[target_feature(enable = "vaes,avx512f")]
		fn __safety_check__(a: __m512i, round_key: __m512i) -> __m512i {
			arch::_mm512_aesdeclast_epi128(a, round_key)
		}
		unsafe { arch::_mm512_aesdeclast_epi128(a, round_key) }
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn aesenc_matches_reference() {
		let Some(((aes, sse2), (vaes, avx512f))) = Aes::try_new()
			.zip(Sse2::try_new())
			.zip(Vaes::try_new().zip(Avx512f::try_new()))
		else {
			return;
		};

		// Same 128-bit block and round key in every lane.
		let a = [0x10u8; 16];
		let k = [0x5Au8; 16];
		let a128 = unsafe { arch::_mm_loadu_si128(a.as_ptr() as *const __m128i) };
		let k128 = unsafe { arch::_mm_loadu_si128(k.as_ptr() as *const __m128i) };
		let mut want = [0u8; 16];
		unsafe {
			sse2._mm_storeu_si128(
				want.as_mut_ptr() as *mut __m128i,
				aes._mm_aesenc_si128(a128, k128),
			);
		}

		let a256 = a.repeat(2);
		let k256 = k.repeat(2);
		let av = unsafe { arch::_mm256_loadu_si256(a256.as_ptr() as *const __m256i) };
		let kv = unsafe { arch::_mm256_loadu_si256(k256.as_ptr() as *const __m256i) };
		let mut got256 = [0u8; 32];
		unsafe {
			arch::_mm256_storeu_si256(
				got256.as_mut_ptr() as *mut __m256i,
				vaes._mm256_aesenc_epi128(av, kv),
			)
		};
		assert_eq!(&got256[..16], want);
		assert_eq!(&got256[16..], want);

		let a512 = a.repeat(4);
		let k512 = k.repeat(4);
		let az = unsafe { arch::_mm512_loadu_si512(a512.as_ptr() as *const _) };
		let kz = unsafe { arch::_mm512_loadu_si512(k512.as_ptr() as *const _) };
		let mut got512 = [0u8; 64];
		unsafe {
			arch::_mm512_storeu_si512(
				got512.as_mut_ptr() as *mut _,
				vaes._mm512_aesenc_epi128(avx512f, az, kz),
			)
		};
		for lane in got512.chunks(16) {
			assert_eq!(lane, want);
		}
	}
}
