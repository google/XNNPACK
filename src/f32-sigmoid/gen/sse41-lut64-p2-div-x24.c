// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/sse-lut64-p2-div.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


extern XNN_INTERNAL const float xnn_table_exp2_k_over_64[64];

void xnn_f32_sigmoid_ukernel__sse41_lut64_p2_div_x24(
    size_t n,
    const float* x,
    float* y,
    const void* params) XNN_DISABLE_TSAN
{
  assert(n % sizeof(float) == 0);

  // Mask for all bits of a floating-point number except the sign bit.
  const __m128 vnonsign_mask = _mm_set1_ps(math_nonsign_mask_f32());

  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p23f);
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const __m128 vdenorm_cutoff = _mm_set1_ps(0x1.5D589Ep+6f);
  const __m128 vminus_log2e_x64 = _mm_set1_ps(-0x1.715476p6f);
  // Last 13 bits are zeroes
  const __m128 vln2_o64_hi = _mm_set1_ps(0x1.630000p-7f);
  const __m128 vln2_o64_lo = _mm_set1_ps(-0x1.BD0106p-19f);
  const __m128 vone = _mm_set1_ps(1.0f);

  const __m128 vc2 = _mm_set1_ps(0x1.FFFF0Ap-2f);

  const __m128i vinv_index_mask = _mm_set1_epi32(~INT32_C(0x3F));

  for (; n >= 24 * sizeof(float); n -= 24 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(x);
    const __m128 vx4567 = _mm_loadu_ps(x + 4);
    const __m128 vx89AB = _mm_loadu_ps(x + 8);
    const __m128 vxCDEF = _mm_loadu_ps(x + 12);
    const __m128 vxGHIJ = _mm_loadu_ps(x + 16);
    const __m128 vxKLMN = _mm_loadu_ps(x + 20);
    x += 24;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const __m128 vz0123 = _mm_and_ps(vx0123, vnonsign_mask);
    const __m128 vz4567 = _mm_and_ps(vx4567, vnonsign_mask);
    const __m128 vz89AB = _mm_and_ps(vx89AB, vnonsign_mask);
    const __m128 vzCDEF = _mm_and_ps(vxCDEF, vnonsign_mask);
    const __m128 vzGHIJ = _mm_and_ps(vxGHIJ, vnonsign_mask);
    const __m128 vzKLMN = _mm_and_ps(vxKLMN, vnonsign_mask);

    // Compute reduced argument n := round(-z * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|z * 64 / log(2)| <= 2**22, i.e.
    // |z| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs x outside of [-87.336544, 17.328678]
    // (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup the result  for such inputs at the
    // very end of the algorithm.
    __m128 vn0123 = _mm_add_ps(vmagic_bias, _mm_mul_ps(vz0123, vminus_log2e_x64));
    __m128 vn4567 = _mm_add_ps(vmagic_bias, _mm_mul_ps(vz4567, vminus_log2e_x64));
    __m128 vn89AB = _mm_add_ps(vmagic_bias, _mm_mul_ps(vz89AB, vminus_log2e_x64));
    __m128 vnCDEF = _mm_add_ps(vmagic_bias, _mm_mul_ps(vzCDEF, vminus_log2e_x64));
    __m128 vnGHIJ = _mm_add_ps(vmagic_bias, _mm_mul_ps(vzGHIJ, vminus_log2e_x64));
    __m128 vnKLMN = _mm_add_ps(vmagic_bias, _mm_mul_ps(vzKLMN, vminus_log2e_x64));

    // Create a floating-point number s (scale) such that s := 2**(n / 64) for such inputs that sigmoidf(-z) is
    // normalized, i.e. 0 <= z <= 87.33642. As n has 6 fractional bits, we split s == 2**(n / 64) =
    // = 2**e * 2**(n / 64 - e), where e := int(n / 64). We create s in two steps:
    // 1. Fetch 2**(n / 64 - e) = 2**(n % 64) from the table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(-z) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 6:14 of n and shift it into bits 23:31 (position of floating-point exponent).
    const __m128i ve0123 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn0123), vinv_index_mask), 17);
    const __m128i ve4567 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn4567), vinv_index_mask), 17);
    const __m128i ve89AB = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn89AB), vinv_index_mask), 17);
    const __m128i veCDEF = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vnCDEF), vinv_index_mask), 17);
    const __m128i veGHIJ = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vnGHIJ), vinv_index_mask), 17);
    const __m128i veKLMN = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vnKLMN), vinv_index_mask), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const __m128i vidx0123 = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vn0123)), 2);
    const __m128i vidx4567 = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vn4567)), 2);
    const __m128i vidx89AB = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vn89AB)), 2);
    const __m128i vidxCDEF = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vnCDEF)), 2);
    const __m128i vidxGHIJ = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vnGHIJ)), 2);
    const __m128i vidxKLMN = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vnKLMN)), 2);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx23 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx0123, vidx0123));
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx01)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx23)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx01 >> 32))), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx23 >> 32))), 1);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx67 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx4567, vidx4567));
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx45)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx67)));
      const __m128i vl45 = _mm_insert_epi32(vl4, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx45 >> 32))), 1);
      const __m128i vl67 = _mm_insert_epi32(vl6, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx67 >> 32))), 1);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
      const uint64_t vidx89 = (uint64_t) _mm_cvtsi128_si64(vidx89AB);
      const uint64_t vidxAB = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx89AB, vidx89AB));
      const __m128i vl8   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx89)));
      const __m128i vlA = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxAB)));
      const __m128i vl89 = _mm_insert_epi32(vl8, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx89 >> 32))), 1);
      const __m128i vlAB = _mm_insert_epi32(vlA, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxAB >> 32))), 1);
      const __m128i vl89AB = _mm_unpacklo_epi64(vl89, vlAB);
      const uint64_t vidxCD = (uint64_t) _mm_cvtsi128_si64(vidxCDEF);
      const uint64_t vidxEF = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidxCDEF, vidxCDEF));
      const __m128i vlC   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxCD)));
      const __m128i vlE = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxEF)));
      const __m128i vlCD = _mm_insert_epi32(vlC, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxCD >> 32))), 1);
      const __m128i vlEF = _mm_insert_epi32(vlE, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxEF >> 32))), 1);
      const __m128i vlCDEF = _mm_unpacklo_epi64(vlCD, vlEF);
      const uint64_t vidxGH = (uint64_t) _mm_cvtsi128_si64(vidxGHIJ);
      const uint64_t vidxIJ = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidxGHIJ, vidxGHIJ));
      const __m128i vlG   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxGH)));
      const __m128i vlI = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxIJ)));
      const __m128i vlGH = _mm_insert_epi32(vlG, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxGH >> 32))), 1);
      const __m128i vlIJ = _mm_insert_epi32(vlI, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxIJ >> 32))), 1);
      const __m128i vlGHIJ = _mm_unpacklo_epi64(vlGH, vlIJ);
      const uint64_t vidxKL = (uint64_t) _mm_cvtsi128_si64(vidxKLMN);
      const uint64_t vidxMN = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidxKLMN, vidxKLMN));
      const __m128i vlK   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxKL)));
      const __m128i vlM = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidxMN)));
      const __m128i vlKL = _mm_insert_epi32(vlK, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxKL >> 32))), 1);
      const __m128i vlMN = _mm_insert_epi32(vlM, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidxMN >> 32))), 1);
      const __m128i vlKLMN = _mm_unpacklo_epi64(vlKL, vlMN);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx0123, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx0123, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx0123, 6);
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx2)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx1)), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx3)), 1);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi16(vidx4567, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi16(vidx4567, 4);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi16(vidx4567, 6);
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx4)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx6)));
      const __m128i vl45 = _mm_insert_epi32(vl4, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx5)), 1);
      const __m128i vl67 = _mm_insert_epi32(vl6, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx7)), 1);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
      const uint32_t vidx8 = (uint32_t) _mm_cvtsi128_si32(vidx89AB);
      const uint32_t vidx9 = (uint32_t) _mm_extract_epi16(vidx89AB, 2);
      const uint32_t vidxA = (uint32_t) _mm_extract_epi16(vidx89AB, 4);
      const uint32_t vidxB = (uint32_t) _mm_extract_epi16(vidx89AB, 6);
      const __m128i vl8   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx8)));
      const __m128i vlA = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxA)));
      const __m128i vl89 = _mm_insert_epi32(vl8, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx9)), 1);
      const __m128i vlAB = _mm_insert_epi32(vlA, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxB)), 1);
      const __m128i vl89AB = _mm_unpacklo_epi64(vl89, vlAB);
      const uint32_t vidxC = (uint32_t) _mm_cvtsi128_si32(vidxCDEF);
      const uint32_t vidxD = (uint32_t) _mm_extract_epi16(vidxCDEF, 2);
      const uint32_t vidxE = (uint32_t) _mm_extract_epi16(vidxCDEF, 4);
      const uint32_t vidxF = (uint32_t) _mm_extract_epi16(vidxCDEF, 6);
      const __m128i vlC   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxC)));
      const __m128i vlE = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxE)));
      const __m128i vlCD = _mm_insert_epi32(vlC, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxD)), 1);
      const __m128i vlEF = _mm_insert_epi32(vlE, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxF)), 1);
      const __m128i vlCDEF = _mm_unpacklo_epi64(vlCD, vlEF);
      const uint32_t vidxG = (uint32_t) _mm_cvtsi128_si32(vidxGHIJ);
      const uint32_t vidxH = (uint32_t) _mm_extract_epi16(vidxGHIJ, 2);
      const uint32_t vidxI = (uint32_t) _mm_extract_epi16(vidxGHIJ, 4);
      const uint32_t vidxJ = (uint32_t) _mm_extract_epi16(vidxGHIJ, 6);
      const __m128i vlG   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxG)));
      const __m128i vlI = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxI)));
      const __m128i vlGH = _mm_insert_epi32(vlG, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxH)), 1);
      const __m128i vlIJ = _mm_insert_epi32(vlI, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxJ)), 1);
      const __m128i vlGHIJ = _mm_unpacklo_epi64(vlGH, vlIJ);
      const uint32_t vidxK = (uint32_t) _mm_cvtsi128_si32(vidxKLMN);
      const uint32_t vidxL = (uint32_t) _mm_extract_epi16(vidxKLMN, 2);
      const uint32_t vidxM = (uint32_t) _mm_extract_epi16(vidxKLMN, 4);
      const uint32_t vidxN = (uint32_t) _mm_extract_epi16(vidxKLMN, 6);
      const __m128i vlK   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxK)));
      const __m128i vlM = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxM)));
      const __m128i vlKL = _mm_insert_epi32(vlK, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxL)), 1);
      const __m128i vlMN = _mm_insert_epi32(vlM, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidxN)), 1);
      const __m128i vlKLMN = _mm_unpacklo_epi64(vlKL, vlMN);
    #endif  // XNN_ARCH_X86_64

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ve0123));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ve4567));
    const __m128 vs89AB = _mm_castsi128_ps(_mm_add_epi32(vl89AB, ve89AB));
    const __m128 vsCDEF = _mm_castsi128_ps(_mm_add_epi32(vlCDEF, veCDEF));
    const __m128 vsGHIJ = _mm_castsi128_ps(_mm_add_epi32(vlGHIJ, veGHIJ));
    const __m128 vsKLMN = _mm_castsi128_ps(_mm_add_epi32(vlKLMN, veKLMN));

    // Subtract the large number back to get the final n := round(-z * 64 / log(2)) as a floating-point number.
    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    vnCDEF = _mm_sub_ps(vnCDEF, vmagic_bias);
    vnGHIJ = _mm_sub_ps(vnGHIJ, vmagic_bias);
    vnKLMN = _mm_sub_ps(vnKLMN, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2) / 64). Note that -t = -z - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note two constants to represent log(2) / 64) to improve accuracy.
    __m128 vt0123 = _mm_add_ps(vz0123, _mm_mul_ps(vn0123, vln2_o64_hi));
    vt0123 = _mm_add_ps(vt0123, _mm_mul_ps(vn0123, vln2_o64_lo));
    __m128 vt4567 = _mm_add_ps(vz4567, _mm_mul_ps(vn4567, vln2_o64_hi));
    vt4567 = _mm_add_ps(vt4567, _mm_mul_ps(vn4567, vln2_o64_lo));
    __m128 vt89AB = _mm_add_ps(vz89AB, _mm_mul_ps(vn89AB, vln2_o64_hi));
    vt89AB = _mm_add_ps(vt89AB, _mm_mul_ps(vn89AB, vln2_o64_lo));
    __m128 vtCDEF = _mm_add_ps(vzCDEF, _mm_mul_ps(vnCDEF, vln2_o64_hi));
    vtCDEF = _mm_add_ps(vtCDEF, _mm_mul_ps(vnCDEF, vln2_o64_lo));
    __m128 vtGHIJ = _mm_add_ps(vzGHIJ, _mm_mul_ps(vnGHIJ, vln2_o64_hi));
    vtGHIJ = _mm_add_ps(vtGHIJ, _mm_mul_ps(vnGHIJ, vln2_o64_lo));
    __m128 vtKLMN = _mm_add_ps(vzKLMN, _mm_mul_ps(vnKLMN, vln2_o64_hi));
    vtKLMN = _mm_add_ps(vtKLMN, _mm_mul_ps(vnKLMN, vln2_o64_lo));

    // Compute degree-2 polynomial approxiatmion for exp(-t) on [-log(2)/128, log(2)/128].
    //   P1(t) = 1 + t * (-1 + t * c2)
    __m128 vp0123 = _mm_mul_ps(vt0123, vc2);
    vp0123 = _mm_sub_ps(vt0123, _mm_mul_ps(vp0123, vt0123));
    __m128 vp4567 = _mm_mul_ps(vt4567, vc2);
    vp4567 = _mm_sub_ps(vt4567, _mm_mul_ps(vp4567, vt4567));
    __m128 vp89AB = _mm_mul_ps(vt89AB, vc2);
    vp89AB = _mm_sub_ps(vt89AB, _mm_mul_ps(vp89AB, vt89AB));
    __m128 vpCDEF = _mm_mul_ps(vtCDEF, vc2);
    vpCDEF = _mm_sub_ps(vtCDEF, _mm_mul_ps(vpCDEF, vtCDEF));
    __m128 vpGHIJ = _mm_mul_ps(vtGHIJ, vc2);
    vpGHIJ = _mm_sub_ps(vtGHIJ, _mm_mul_ps(vpGHIJ, vtGHIJ));
    __m128 vpKLMN = _mm_mul_ps(vtKLMN, vc2);
    vpKLMN = _mm_sub_ps(vtKLMN, _mm_mul_ps(vpKLMN, vtKLMN));

    // Reconstruct the exp(-z) value:
    //   f = s * (1 + t * (-1 + t * c2))
    //     = s * (1 - t + t * (t * c2))
    //     = s - s * (t - t * (t * c2))
    //     = s - s * p
    const __m128 vy0123 = _mm_sub_ps(vs0123, _mm_mul_ps(vs0123, vp0123));
    const __m128 vy4567 = _mm_sub_ps(vs4567, _mm_mul_ps(vs4567, vp4567));
    const __m128 vy89AB = _mm_sub_ps(vs89AB, _mm_mul_ps(vs89AB, vp89AB));
    const __m128 vyCDEF = _mm_sub_ps(vsCDEF, _mm_mul_ps(vsCDEF, vpCDEF));
    const __m128 vyGHIJ = _mm_sub_ps(vsGHIJ, _mm_mul_ps(vsGHIJ, vpGHIJ));
    const __m128 vyKLMN = _mm_sub_ps(vsKLMN, _mm_mul_ps(vsKLMN, vpKLMN));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    __m128 vf0123 = _mm_div_ps(vy0123, _mm_add_ps(vy0123, vone));
    __m128 vf4567 = _mm_div_ps(vy4567, _mm_add_ps(vy4567, vone));
    __m128 vf89AB = _mm_div_ps(vy89AB, _mm_add_ps(vy89AB, vone));
    __m128 vfCDEF = _mm_div_ps(vyCDEF, _mm_add_ps(vyCDEF, vone));
    __m128 vfGHIJ = _mm_div_ps(vyGHIJ, _mm_add_ps(vyGHIJ, vone));
    __m128 vfKLMN = _mm_div_ps(vyKLMN, _mm_add_ps(vyKLMN, vone));

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vz0123), vf0123);
    vf4567 = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vz4567), vf4567);
    vf89AB = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vz89AB), vf89AB);
    vfCDEF = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vzCDEF), vfCDEF);
    vfGHIJ = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vzGHIJ), vfGHIJ);
    vfKLMN = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vzKLMN), vfKLMN);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf0123 = _mm_blendv_ps(_mm_sub_ps(vone, vf0123), vf0123, vx0123);
    vf4567 = _mm_blendv_ps(_mm_sub_ps(vone, vf4567), vf4567, vx4567);
    vf89AB = _mm_blendv_ps(_mm_sub_ps(vone, vf89AB), vf89AB, vx89AB);
    vfCDEF = _mm_blendv_ps(_mm_sub_ps(vone, vfCDEF), vfCDEF, vxCDEF);
    vfGHIJ = _mm_blendv_ps(_mm_sub_ps(vone, vfGHIJ), vfGHIJ, vxGHIJ);
    vfKLMN = _mm_blendv_ps(_mm_sub_ps(vone, vfKLMN), vfKLMN, vxKLMN);

    _mm_storeu_ps(y, vf0123);
    _mm_storeu_ps(y + 4, vf4567);
    _mm_storeu_ps(y + 8, vf89AB);
    _mm_storeu_ps(y + 12, vfCDEF);
    _mm_storeu_ps(y + 16, vfGHIJ);
    _mm_storeu_ps(y + 20, vfKLMN);
    y += 24;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(x);
    x += 4;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const __m128 vz = _mm_and_ps(vx, vnonsign_mask);

    // Compute reduced argument n := round(-z * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|z * 64 / log(2)| <= 2**22, i.e.
    // |z| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs x outside of [-87.336544, 17.328678]
    // (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup the result  for such inputs at the
    // very end of the algorithm.
    __m128 vn = _mm_add_ps(vmagic_bias, _mm_mul_ps(vz, vminus_log2e_x64));

    // Create a floating-point number s (scale) such that s := 2**(n / 64) for such inputs that sigmoidf(-z) is
    // normalized, i.e. 0 <= z <= 87.33642. As n has 6 fractional bits, we split s == 2**(n / 64) =
    // = 2**e * 2**(n / 64 - e), where e := int(n / 64). We create s in two steps:
    // 1. Fetch 2**(n / 64 - e) = 2**(n % 64) from the table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(-z) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 6:14 of n and shift it into bits 23:31 (position of floating-point exponent).
    const __m128i ve = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vinv_index_mask), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const __m128i vidx = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vn)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl0 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx_lo)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx_hi)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx_lo >> 32))), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx_hi >> 32))), 1);
      const __m128i vl = _mm_unpacklo_epi64(vl01, vl23);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      const __m128i vl0 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx2)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx1)), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx3)), 1);
      const __m128i vl = _mm_unpacklo_epi64(vl01, vl23);
    #endif  // XNN_ARCH_X86_64
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    // Subtract the large number back to get the final n := round(-z * 64 / log(2)) as a floating-point number.
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2) / 64). Note that -t = -z - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note two constants to represent log(2) / 64) to improve accuracy.
    __m128 vt = _mm_add_ps(vz, _mm_mul_ps(vn, vln2_o64_hi));
    vt = _mm_add_ps(vt, _mm_mul_ps(vn, vln2_o64_lo));

    // Compute degree-2 polynomial approxiatmion for exp(-t) on [-log(2)/128, log(2)/128].
    //   P1(t) = 1 + t * (-1 + t * c2)
    __m128 vp = _mm_mul_ps(vt, vc2);
    vp = _mm_sub_ps(vt, _mm_mul_ps(vp, vt));

    // Reconstruct the exp(-z) value:
    //   f = s * (1 + t * (-1 + t * c2))
    //     = s * (1 - t + t * (t * c2))
    //     = s - s * (t - t * (t * c2))
    //     = s - s * p
    const __m128 vy = _mm_sub_ps(vs, _mm_mul_ps(vs, vp));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    __m128 vf = _mm_div_ps(vy, _mm_add_ps(vy, vone));

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vz), vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = _mm_blendv_ps(_mm_sub_ps(vone, vf), vf, vx);

    _mm_storeu_ps(y, vf);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m128 vx = _mm_loadu_ps(x);

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const __m128 vz = _mm_and_ps(vx, vnonsign_mask);

    // Compute reduced argument n := round(-z * 64 / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to an integer, then subtracing
    // the large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|z * 64 / log(2)| <= 2**22, i.e.
    // |z| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs x outside of [-87.336544, 17.328678]
    // (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup the result  for such inputs at the
    // very end of the algorithm.
    __m128 vn = _mm_add_ps(vmagic_bias, _mm_mul_ps(vz, vminus_log2e_x64));

    // Create a floating-point number s (scale) such that s := 2**(n / 64) for such inputs that sigmoidf(-z) is
    // normalized, i.e. 0 <= z <= 87.33642. As n has 6 fractional bits, we split s == 2**(n / 64) =
    // = 2**e * 2**(n / 64 - e), where e := int(n / 64). We create s in two steps:
    // 1. Fetch 2**(n / 64 - e) = 2**(n % 64) from the table using the 6 low bits of n, as integer. Note that the
    //    fetched values are in the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of e to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 87.33642 (inputs for which sigmoidf(-z) is normalized) we have -126 <= e <= 0,
    //    and thus the adjusted exponent is not lower than -126.
    //
    // Extract e from bits 6:14 of n and shift it into bits 23:31 (position of floating-point exponent).
    const __m128i ve = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vinv_index_mask), 17);

    // Use bits 0:6 bits of n, as integer, as an index for table lookup of l := 2**(n % 64).
    const __m128i vidx = _mm_slli_epi32(_mm_andnot_si128(vinv_index_mask, _mm_castps_si128(vn)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl0 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx_lo)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) vidx_hi)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx_lo >> 32))), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + (uint32_t) (vidx_hi >> 32))), 1);
      const __m128i vl = _mm_unpacklo_epi64(vl01, vl23);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      const __m128i vl0 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx2)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx1)), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2_k_over_64 + vidx3)), 1);
      const __m128i vl = _mm_unpacklo_epi64(vl01, vl23);
    #endif  // XNN_ARCH_X86_64

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    // Subtract the large number back to get the final n := round(-z * 64 / log(2)) as a floating-point number.
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := (z + n * log(2) / 64). Note that -t = -z - n * log(2) / 64.
    // Use Cody-Waite range reduction method (note two constants to represent log(2) / 64) to improve accuracy.
    __m128 vt = _mm_add_ps(vz, _mm_mul_ps(vn, vln2_o64_hi));
    vt = _mm_add_ps(vt, _mm_mul_ps(vn, vln2_o64_lo));

    // Compute degree-2 polynomial approxiatmion for exp(-t) on [-log(2)/128, log(2)/128].
    //   P1(t) = 1 + t * (-1 + t * c2)
    __m128 vp = _mm_mul_ps(vt, vc2);
    vp = _mm_sub_ps(vt, _mm_mul_ps(vp, vt));

    // Reconstruct the exp(-z) value:
    //   f = s * (1 + t * (-1 + t * c2))
    //     = s * (1 - t + t * (t * c2))
    //     = s - s * (t - t * (t * c2))
    //     = s - s * p
    const __m128 vy = _mm_sub_ps(vs, _mm_mul_ps(vs, vp));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    __m128 vf = _mm_div_ps(vy, _mm_add_ps(vy, vone));

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vdenorm_cutoff, vz), vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = _mm_blendv_ps(_mm_sub_ps(vone, vf), vf, vx);

    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vf);
      vf = _mm_movehl_ps(vf, vf);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vf);
    }
  }
}
