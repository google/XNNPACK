// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx-expm1minus-lut.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.sign_mask);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.sat_cutoff);
  const __m256 vlog2e = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.magic_bias);
  const __m128i vindex_mask = _mm_load_si128((const __m128i*) params->avx_expm1minus_rr1_lut8_p4h3.index_mask);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.minus_ln2);
  const __m256 vc4 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.c2);
  const __m256 vtwo = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.two);
  const __m256 vminus_one = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.minus_one);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(input);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(input + 8);
    input += 16;

    __m256 vz01234567 = _mm256_or_ps(vx01234567, vsign_mask);
    __m256 vz89ABCDEF = _mm256_or_ps(vx89ABCDEF, vsign_mask);

    const __m256 vinvsignx01234567 = _mm256_xor_ps(vx01234567, vz01234567);
    vz01234567 = _mm256_max_ps(vsat_cutoff, vz01234567);
    const __m256 vinvsignx89ABCDEF = _mm256_xor_ps(vx89ABCDEF, vz89ABCDEF);
    vz89ABCDEF = _mm256_max_ps(vsat_cutoff, vz89ABCDEF);

    __m256 vn01234567 = _mm256_add_ps(_mm256_mul_ps(vz01234567, vlog2e), vmagic_bias);
    __m256 vn89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vz89ABCDEF, vlog2e), vmagic_bias);

    const __m128 vn4567 = _mm256_extractf128_ps(vn01234567, 1);
    const __m128i ve0123 = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn01234567)), 20);
    const __m128 vnCDEF = _mm256_extractf128_ps(vn89ABCDEF, 1);
    const __m128i ve89AB = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn89ABCDEF)), 20);

    const __m128i ve4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 20);
    const __m128i vidx0123 = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn01234567)), vindex_mask);
    const __m128i vidx4567 = _mm_and_si128(_mm_castps_si128(vn4567), vindex_mask);
    const __m128i veCDEF = _mm_slli_epi32(_mm_castps_si128(vnCDEF), 20);
    const __m128i vidx89AB = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn89ABCDEF)), vindex_mask);
    const __m128i vidxCDEF = _mm_and_si128(_mm_castps_si128(vnCDEF), vindex_mask);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx89 = (uint64_t) _mm_cvtsi128_si64(vidx89AB);
      const uint64_t vidxCD = (uint64_t) _mm_cvtsi128_si64(vidxCDEF);

      __m128i vl0123 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl4567 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      __m128i vl89AB = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx89]);
      __m128i vlCDEF = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxCD]);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx0123, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx4567, 1);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx89 >> 32)], 1);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxCD >> 32)], 1);
      const uint64_t vidxAB = (uint64_t) _mm_extract_epi64(vidx89AB, 1);
      const uint64_t vidxEF = (uint64_t) _mm_extract_epi64(vidxCDEF, 1);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxAB], 2);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxEF], 2);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxAB >> 32)], 3);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxEF >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx8 = (uint32_t) _mm_cvtsi128_si32(vidx89AB);
      const uint32_t vidxC = (uint32_t) _mm_cvtsi128_si32(vidxCDEF);

      __m128i vl0123 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl4567 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx0123, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx4567, 1);
      __m128i vl89AB = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx8]);
      __m128i vlCDEF = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxC]);
      const uint32_t vidx9 = (uint32_t) _mm_extract_epi32(vidx89AB, 1);
      const uint32_t vidxD = (uint32_t) _mm_extract_epi32(vidxCDEF, 1);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx0123, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx4567, 2);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx9], 1);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxD], 1);
      const uint32_t vidxA = (uint32_t) _mm_extract_epi32(vidx89AB, 2);
      const uint32_t vidxE = (uint32_t) _mm_extract_epi32(vidxCDEF, 2);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx0123, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx4567, 3);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxA], 2);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxE], 2);
      const uint32_t vidxB = (uint32_t) _mm_extract_epi32(vidx89AB, 3);
      const uint32_t vidxF = (uint32_t) _mm_extract_epi32(vidxCDEF, 3);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxB], 3);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxF], 3);
    #endif

    const __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ve0123));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ve4567));
    const __m128 vs89AB = _mm_castsi128_ps(_mm_add_epi32(vl89AB, ve89AB));
    const __m128 vsCDEF = _mm_castsi128_ps(_mm_add_epi32(vlCDEF, veCDEF));

    const __m256 vs01234567 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs0123), vs4567, 1);
    vn01234567 = _mm256_sub_ps(vn01234567, vmagic_bias);
    const __m256 vs89ABCDEF = _mm256_insertf128_ps(_mm256_castps128_ps256(vs89AB), vsCDEF, 1);
    vn89ABCDEF = _mm256_sub_ps(vn89ABCDEF, vmagic_bias);

    const __m256 vt01234567 = _mm256_add_ps(_mm256_mul_ps(vn01234567, vminus_ln2), vz01234567);
    const __m256 vt89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vn89ABCDEF, vminus_ln2), vz89ABCDEF);

    __m256 vp01234567 = _mm256_add_ps(_mm256_mul_ps(vc4, vt01234567), vc3);
    __m256 vp89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vc4, vt89ABCDEF), vc3);
    vp01234567 = _mm256_add_ps(_mm256_mul_ps(vp01234567, vt01234567), vc2);
    vp89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vp89ABCDEF, vt89ABCDEF), vc2);
    vp01234567 = _mm256_add_ps(_mm256_mul_ps(vp01234567, vt01234567), vtwo);
    vp89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vp89ABCDEF, vt89ABCDEF), vtwo);

    const __m256 vts01234567 = _mm256_mul_ps(vt01234567, vs01234567);
    const __m256 vsmo01234567 = _mm256_add_ps(vs01234567, vminus_one);
    const __m256 vts89ABCDEF = _mm256_mul_ps(vt89ABCDEF, vs89ABCDEF);
    const __m256 vsmo89ABCDEF = _mm256_add_ps(vs89ABCDEF, vminus_one);
    const __m256 vemo01234567 = _mm256_add_ps(_mm256_mul_ps(vp01234567, vts01234567), vsmo01234567);
    const __m256 vemo89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vp89ABCDEF, vts89ABCDEF), vsmo89ABCDEF);
    const __m256 vepo01234567 = _mm256_add_ps(vemo01234567, vtwo);
    const __m256 vepo89ABCDEF = _mm256_add_ps(vemo89ABCDEF, vtwo);

    __m256 vy01234567 = _mm256_div_ps(vemo01234567, vepo01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(vemo89ABCDEF, vepo89ABCDEF);

    vy01234567 = _mm256_xor_ps(vy01234567, vinvsignx01234567);
    vy89ABCDEF = _mm256_xor_ps(vy89ABCDEF, vinvsignx89ABCDEF);

    _mm256_storeu_ps(output, vy01234567);
    _mm256_storeu_ps(output + 8, vy89ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    __m256 vz = _mm256_or_ps(vx, vsign_mask);
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    const __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 20);
    const __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 20);

    const __m128i vidx_lo = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn)), vindex_mask);
    const __m128i vidx_hi = _mm_and_si128(_mm_castps_si128(vn_hi), vindex_mask);
    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx_lo);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx_lo, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx_lo, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx_hi, 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx_lo, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx_hi, 3);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
    #endif

    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ve_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ve_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vp, vts), vsmo);
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vy = _mm256_div_ps(vemo, vepo);

    vy = _mm256_xor_ps(vy, vinvsignx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_expm1minus_rr1_lut8_p4h3.mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    __m256 vz = _mm256_or_ps(vx, vsign_mask);
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    const __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 20);
    const __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 20);

    const __m128i vidx_lo = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn)), vindex_mask);
    const __m128i vidx_hi = _mm_and_si128(_mm_castps_si128(vn_hi), vindex_mask);
    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx_lo);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx_lo, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx_lo, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx_hi, 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx_lo, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx_hi, 3);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
    #endif

    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ve_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ve_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc4, vt), vc3);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vp, vts), vsmo);
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vy = _mm256_div_ps(vemo, vepo);

    vy = _mm256_xor_ps(vy, vinvsignx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
