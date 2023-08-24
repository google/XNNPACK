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

void xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8(
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

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    __m256 vz = _mm256_or_ps(vx, vsign_mask);
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

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

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc4;
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);
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

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

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

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc4;
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);
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
