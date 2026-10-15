use super::*;

impl Aes {
	delegate!(
		#[target_feature(enable = "aes")]
		{
			fn _mm_aesenc_si128(a: __m128i, round_key: __m128i) -> __m128i;
			fn _mm_aesenclast_si128(a: __m128i, round_key: __m128i) -> __m128i;
			fn _mm_aesdec_si128(a: __m128i, round_key: __m128i) -> __m128i;
			fn _mm_aesdeclast_si128(a: __m128i, round_key: __m128i) -> __m128i;
			fn _mm_aesimc_si128(a: __m128i) -> __m128i;
			fn _mm_aeskeygenassist_si128<const IMM8: i32>(a: __m128i) -> __m128i;
		}
	);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn aesenc_matches_reference() {
		if Aes::try_new().is_none() {
			return;
		}
		let aes = Aes::try_new().unwrap();
		let a = unsafe {
			arch::_mm_set_epi64x(0x0123456789ABCDEFu64 as i64, 0xFEDCBA9876543210u64 as i64)
		};
		let k = unsafe { arch::_mm_set_epi64x(0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F) };
		let mut got = [0u8; 16];
		let mut want = [0u8; 16];
		unsafe {
			arch::_mm_storeu_si128(got.as_mut_ptr() as *mut __m128i, aes._mm_aesenc_si128(a, k));
			arch::_mm_storeu_si128(
				want.as_mut_ptr() as *mut __m128i,
				arch::_mm_aesenc_si128(a, k),
			);
		}
		assert_eq!(got, want);
	}
}
