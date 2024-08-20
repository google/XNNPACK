// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx-rr2-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u48(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.154246p+4f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p19f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vindex_mask = _mm256_castsi256_ps(_mm256_set1_epi32(UINT32_C(0xF)));
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E400p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(-0x1.7F7D1Cp-20f);
  const __m256 vc3 = _mm256_set1_ps(0x1.55561Cp-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.0001ECp-1f);
  const __m256 vone = _mm256_set1_ps(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);
  
  const __m256 vprescale = _mm256_set1_ps(params->scalar.prescale);
  const __m256 valpha = _mm256_set1_ps(params->scalar.alpha);
  const __m256 vbeta = _mm256_set1_ps(params->scalar.beta);

  for (; batch >= 48 * sizeof(float); batch -= 48 * sizeof(float)) {
    __m256 vx0 = _mm256_loadu_ps(input);
    __m256 vx1 = _mm256_loadu_ps(input + 8);
    __m256 vx2 = _mm256_loadu_ps(input + 16);
    __m256 vx3 = _mm256_loadu_ps(input + 24);
    __m256 vx4 = _mm256_loadu_ps(input + 32);
    __m256 vx5 = _mm256_loadu_ps(input + 40);
    input += 48;

    const __m256 vz0 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx0, vprescale));
    const __m256 vz1 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx1, vprescale));
    const __m256 vz2 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx2, vprescale));
    const __m256 vz3 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx3, vprescale));
    const __m256 vz4 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx4, vprescale));
    const __m256 vz5 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx5, vprescale));

    __m256 vn0 = _mm256_add_ps(_mm256_mul_ps(vz0, vlog2e), vmagic_bias);
    __m256 vn1 = _mm256_add_ps(_mm256_mul_ps(vz1, vlog2e), vmagic_bias);
    __m256 vn2 = _mm256_add_ps(_mm256_mul_ps(vz2, vlog2e), vmagic_bias);
    __m256 vn3 = _mm256_add_ps(_mm256_mul_ps(vz3, vlog2e), vmagic_bias);
    __m256 vn4 = _mm256_add_ps(_mm256_mul_ps(vz4, vlog2e), vmagic_bias);
    __m256 vn5 = _mm256_add_ps(_mm256_mul_ps(vz5, vlog2e), vmagic_bias);

    const __m256 vidx0 = _mm256_and_ps(vn0, vindex_mask);

    const __m128i vidx0_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx0)), 2);
    const __m128i vidx0_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx0, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx0_ll = (uint64_t) _mm_cvtsi128_si64(vidx0_lo);
      const uint64_t vidx0_lh = (uint64_t) _mm_extract_epi64(vidx0_lo, 1);
      const uint64_t vidx0_hl = (uint64_t) _mm_cvtsi128_si64(vidx0_hi);
      const uint64_t vidx0_hh = (uint64_t) _mm_extract_epi64(vidx0_hi, 1);
      __m128i vl0_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0_ll)));
      __m128i vl0_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0_lh)));
      __m128i vl0_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0_hl)));
      __m128i vl0_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx0_hh)));
      vl0_ll = _mm_insert_epi32(vl0_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx0_ll >> 32))), 1);
      vl0_lh = _mm_insert_epi32(vl0_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx0_lh >> 32))), 1);
      vl0_hl = _mm_insert_epi32(vl0_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx0_hl >> 32))), 1);
      vl0_hh = _mm_insert_epi32(vl0_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx0_hh >> 32))), 1);
    #else
      __m128i vl0_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx0_lo))));
      __m128i vl0_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx0_lo, 2))));
      __m128i vl0_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx0_hi))));
      __m128i vl0_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx0_hi, 2))));
      vl0_ll = _mm_insert_epi32(vl0_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx0_lo, 1))), 1);
      vl0_lh = _mm_insert_epi32(vl0_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx0_lo, 3))), 1);
      vl0_hl = _mm_insert_epi32(vl0_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx0_hi, 1))), 1);
      vl0_hh = _mm_insert_epi32(vl0_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx0_hi, 3))), 1);
    #endif
    const __m128i vl0_lo = _mm_unpacklo_epi64(vl0_ll, vl0_lh);
    const __m128i vl0_hi = _mm_unpacklo_epi64(vl0_hl, vl0_hh);
    const __m256 vidx1 = _mm256_and_ps(vn1, vindex_mask);

    const __m128i vidx1_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx1)), 2);
    const __m128i vidx1_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx1, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx1_ll = (uint64_t) _mm_cvtsi128_si64(vidx1_lo);
      const uint64_t vidx1_lh = (uint64_t) _mm_extract_epi64(vidx1_lo, 1);
      const uint64_t vidx1_hl = (uint64_t) _mm_cvtsi128_si64(vidx1_hi);
      const uint64_t vidx1_hh = (uint64_t) _mm_extract_epi64(vidx1_hi, 1);
      __m128i vl1_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1_ll)));
      __m128i vl1_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1_lh)));
      __m128i vl1_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1_hl)));
      __m128i vl1_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx1_hh)));
      vl1_ll = _mm_insert_epi32(vl1_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx1_ll >> 32))), 1);
      vl1_lh = _mm_insert_epi32(vl1_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx1_lh >> 32))), 1);
      vl1_hl = _mm_insert_epi32(vl1_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx1_hl >> 32))), 1);
      vl1_hh = _mm_insert_epi32(vl1_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx1_hh >> 32))), 1);
    #else
      __m128i vl1_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx1_lo))));
      __m128i vl1_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx1_lo, 2))));
      __m128i vl1_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx1_hi))));
      __m128i vl1_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx1_hi, 2))));
      vl1_ll = _mm_insert_epi32(vl1_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx1_lo, 1))), 1);
      vl1_lh = _mm_insert_epi32(vl1_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx1_lo, 3))), 1);
      vl1_hl = _mm_insert_epi32(vl1_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx1_hi, 1))), 1);
      vl1_hh = _mm_insert_epi32(vl1_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx1_hi, 3))), 1);
    #endif
    const __m128i vl1_lo = _mm_unpacklo_epi64(vl1_ll, vl1_lh);
    const __m128i vl1_hi = _mm_unpacklo_epi64(vl1_hl, vl1_hh);
    const __m256 vidx2 = _mm256_and_ps(vn2, vindex_mask);

    const __m128i vidx2_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx2)), 2);
    const __m128i vidx2_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx2, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx2_ll = (uint64_t) _mm_cvtsi128_si64(vidx2_lo);
      const uint64_t vidx2_lh = (uint64_t) _mm_extract_epi64(vidx2_lo, 1);
      const uint64_t vidx2_hl = (uint64_t) _mm_cvtsi128_si64(vidx2_hi);
      const uint64_t vidx2_hh = (uint64_t) _mm_extract_epi64(vidx2_hi, 1);
      __m128i vl2_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2_ll)));
      __m128i vl2_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2_lh)));
      __m128i vl2_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2_hl)));
      __m128i vl2_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx2_hh)));
      vl2_ll = _mm_insert_epi32(vl2_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx2_ll >> 32))), 1);
      vl2_lh = _mm_insert_epi32(vl2_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx2_lh >> 32))), 1);
      vl2_hl = _mm_insert_epi32(vl2_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx2_hl >> 32))), 1);
      vl2_hh = _mm_insert_epi32(vl2_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx2_hh >> 32))), 1);
    #else
      __m128i vl2_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx2_lo))));
      __m128i vl2_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx2_lo, 2))));
      __m128i vl2_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx2_hi))));
      __m128i vl2_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx2_hi, 2))));
      vl2_ll = _mm_insert_epi32(vl2_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx2_lo, 1))), 1);
      vl2_lh = _mm_insert_epi32(vl2_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx2_lo, 3))), 1);
      vl2_hl = _mm_insert_epi32(vl2_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx2_hi, 1))), 1);
      vl2_hh = _mm_insert_epi32(vl2_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx2_hi, 3))), 1);
    #endif
    const __m128i vl2_lo = _mm_unpacklo_epi64(vl2_ll, vl2_lh);
    const __m128i vl2_hi = _mm_unpacklo_epi64(vl2_hl, vl2_hh);
    const __m256 vidx3 = _mm256_and_ps(vn3, vindex_mask);

    const __m128i vidx3_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx3)), 2);
    const __m128i vidx3_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx3, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx3_ll = (uint64_t) _mm_cvtsi128_si64(vidx3_lo);
      const uint64_t vidx3_lh = (uint64_t) _mm_extract_epi64(vidx3_lo, 1);
      const uint64_t vidx3_hl = (uint64_t) _mm_cvtsi128_si64(vidx3_hi);
      const uint64_t vidx3_hh = (uint64_t) _mm_extract_epi64(vidx3_hi, 1);
      __m128i vl3_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3_ll)));
      __m128i vl3_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3_lh)));
      __m128i vl3_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3_hl)));
      __m128i vl3_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx3_hh)));
      vl3_ll = _mm_insert_epi32(vl3_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx3_ll >> 32))), 1);
      vl3_lh = _mm_insert_epi32(vl3_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx3_lh >> 32))), 1);
      vl3_hl = _mm_insert_epi32(vl3_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx3_hl >> 32))), 1);
      vl3_hh = _mm_insert_epi32(vl3_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx3_hh >> 32))), 1);
    #else
      __m128i vl3_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx3_lo))));
      __m128i vl3_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx3_lo, 2))));
      __m128i vl3_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx3_hi))));
      __m128i vl3_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx3_hi, 2))));
      vl3_ll = _mm_insert_epi32(vl3_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx3_lo, 1))), 1);
      vl3_lh = _mm_insert_epi32(vl3_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx3_lo, 3))), 1);
      vl3_hl = _mm_insert_epi32(vl3_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx3_hi, 1))), 1);
      vl3_hh = _mm_insert_epi32(vl3_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx3_hi, 3))), 1);
    #endif
    const __m128i vl3_lo = _mm_unpacklo_epi64(vl3_ll, vl3_lh);
    const __m128i vl3_hi = _mm_unpacklo_epi64(vl3_hl, vl3_hh);
    const __m256 vidx4 = _mm256_and_ps(vn4, vindex_mask);

    const __m128i vidx4_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx4)), 2);
    const __m128i vidx4_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx4, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx4_ll = (uint64_t) _mm_cvtsi128_si64(vidx4_lo);
      const uint64_t vidx4_lh = (uint64_t) _mm_extract_epi64(vidx4_lo, 1);
      const uint64_t vidx4_hl = (uint64_t) _mm_cvtsi128_si64(vidx4_hi);
      const uint64_t vidx4_hh = (uint64_t) _mm_extract_epi64(vidx4_hi, 1);
      __m128i vl4_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx4_ll)));
      __m128i vl4_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx4_lh)));
      __m128i vl4_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx4_hl)));
      __m128i vl4_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx4_hh)));
      vl4_ll = _mm_insert_epi32(vl4_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx4_ll >> 32))), 1);
      vl4_lh = _mm_insert_epi32(vl4_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx4_lh >> 32))), 1);
      vl4_hl = _mm_insert_epi32(vl4_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx4_hl >> 32))), 1);
      vl4_hh = _mm_insert_epi32(vl4_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx4_hh >> 32))), 1);
    #else
      __m128i vl4_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx4_lo))));
      __m128i vl4_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx4_lo, 2))));
      __m128i vl4_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx4_hi))));
      __m128i vl4_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx4_hi, 2))));
      vl4_ll = _mm_insert_epi32(vl4_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx4_lo, 1))), 1);
      vl4_lh = _mm_insert_epi32(vl4_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx4_lo, 3))), 1);
      vl4_hl = _mm_insert_epi32(vl4_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx4_hi, 1))), 1);
      vl4_hh = _mm_insert_epi32(vl4_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx4_hi, 3))), 1);
    #endif
    const __m128i vl4_lo = _mm_unpacklo_epi64(vl4_ll, vl4_lh);
    const __m128i vl4_hi = _mm_unpacklo_epi64(vl4_hl, vl4_hh);
    const __m256 vidx5 = _mm256_and_ps(vn5, vindex_mask);

    const __m128i vidx5_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx5)), 2);
    const __m128i vidx5_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx5, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx5_ll = (uint64_t) _mm_cvtsi128_si64(vidx5_lo);
      const uint64_t vidx5_lh = (uint64_t) _mm_extract_epi64(vidx5_lo, 1);
      const uint64_t vidx5_hl = (uint64_t) _mm_cvtsi128_si64(vidx5_hi);
      const uint64_t vidx5_hh = (uint64_t) _mm_extract_epi64(vidx5_hi, 1);
      __m128i vl5_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx5_ll)));
      __m128i vl5_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx5_lh)));
      __m128i vl5_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx5_hl)));
      __m128i vl5_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx5_hh)));
      vl5_ll = _mm_insert_epi32(vl5_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx5_ll >> 32))), 1);
      vl5_lh = _mm_insert_epi32(vl5_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx5_lh >> 32))), 1);
      vl5_hl = _mm_insert_epi32(vl5_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx5_hl >> 32))), 1);
      vl5_hh = _mm_insert_epi32(vl5_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx5_hh >> 32))), 1);
    #else
      __m128i vl5_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx5_lo))));
      __m128i vl5_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx5_lo, 2))));
      __m128i vl5_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx5_hi))));
      __m128i vl5_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx5_hi, 2))));
      vl5_ll = _mm_insert_epi32(vl5_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx5_lo, 1))), 1);
      vl5_lh = _mm_insert_epi32(vl5_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx5_lo, 3))), 1);
      vl5_hl = _mm_insert_epi32(vl5_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx5_hi, 1))), 1);
      vl5_hh = _mm_insert_epi32(vl5_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx5_hi, 3))), 1);
    #endif
    const __m128i vl5_lo = _mm_unpacklo_epi64(vl5_ll, vl5_lh);
    const __m128i vl5_hi = _mm_unpacklo_epi64(vl5_hl, vl5_hh);

    const __m128i ven0_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn0)), 19);
    const __m128i ven0_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn0, 1)), 19);
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    const __m128 vs0_lo = _mm_castsi128_ps(_mm_add_epi32(vl0_lo, ven0_lo));
    const __m128 vs0_hi = _mm_castsi128_ps(_mm_add_epi32(vl0_hi, ven0_hi));
    const __m128i ven1_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn1)), 19);
    const __m128i ven1_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn1, 1)), 19);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    const __m128 vs1_lo = _mm_castsi128_ps(_mm_add_epi32(vl1_lo, ven1_lo));
    const __m128 vs1_hi = _mm_castsi128_ps(_mm_add_epi32(vl1_hi, ven1_hi));
    const __m128i ven2_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn2)), 19);
    const __m128i ven2_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn2, 1)), 19);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    const __m128 vs2_lo = _mm_castsi128_ps(_mm_add_epi32(vl2_lo, ven2_lo));
    const __m128 vs2_hi = _mm_castsi128_ps(_mm_add_epi32(vl2_hi, ven2_hi));
    const __m128i ven3_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn3)), 19);
    const __m128i ven3_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn3, 1)), 19);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    const __m128 vs3_lo = _mm_castsi128_ps(_mm_add_epi32(vl3_lo, ven3_lo));
    const __m128 vs3_hi = _mm_castsi128_ps(_mm_add_epi32(vl3_hi, ven3_hi));
    const __m128i ven4_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn4)), 19);
    const __m128i ven4_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn4, 1)), 19);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    const __m128 vs4_lo = _mm_castsi128_ps(_mm_add_epi32(vl4_lo, ven4_lo));
    const __m128 vs4_hi = _mm_castsi128_ps(_mm_add_epi32(vl4_hi, ven4_hi));
    const __m128i ven5_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn5)), 19);
    const __m128i ven5_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn5, 1)), 19);
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    const __m128 vs5_lo = _mm_castsi128_ps(_mm_add_epi32(vl5_lo, ven5_lo));
    const __m128 vs5_hi = _mm_castsi128_ps(_mm_add_epi32(vl5_hi, ven5_hi));

    __m256 vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_hi), vz0);
    __m256 vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_hi), vz1);
    __m256 vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_hi), vz2);
    __m256 vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_hi), vz3);
    __m256 vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2_hi), vz4);
    __m256 vt5 = _mm256_add_ps(_mm256_mul_ps(vn5, vminus_ln2_hi), vz5);

    vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2_lo), vt0);
    __m256 vs0 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs0_lo), vs0_hi, 1);
    vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2_lo), vt1);
    __m256 vs1 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs1_lo), vs1_hi, 1);
    vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2_lo), vt2);
    __m256 vs2 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs2_lo), vs2_hi, 1);
    vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2_lo), vt3);
    __m256 vs3 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs3_lo), vs3_hi, 1);
    vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2_lo), vt4);
    __m256 vs4 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs4_lo), vs4_hi, 1);
    vt5 = _mm256_add_ps(_mm256_mul_ps(vn5, vminus_ln2_lo), vt5);
    __m256 vs5 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs5_lo), vs5_hi, 1);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc3, vt0), vc2);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc3, vt1), vc2);
    __m256 vp2 = _mm256_add_ps(_mm256_mul_ps(vc3, vt2), vc2);
    __m256 vp3 = _mm256_add_ps(_mm256_mul_ps(vc3, vt3), vc2);
    __m256 vp4 = _mm256_add_ps(_mm256_mul_ps(vc3, vt4), vc2);
    __m256 vp5 = _mm256_add_ps(_mm256_mul_ps(vc3, vt5), vc2);

    vp0 = _mm256_mul_ps(vp0, vt0);
    vp1 = _mm256_mul_ps(vp1, vt1);
    vp2 = _mm256_mul_ps(vp2, vt2);
    vp3 = _mm256_mul_ps(vp3, vt3);
    vp4 = _mm256_mul_ps(vp4, vt4);
    vp5 = _mm256_mul_ps(vp5, vt5);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vs0 = _mm256_sub_ps(vs0, vone);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vs1 = _mm256_sub_ps(vs1, vone);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vs2 = _mm256_sub_ps(vs2, vone);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vs3 = _mm256_sub_ps(vs3, vone);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vs4 = _mm256_sub_ps(vs4, vone);
    vt5 = _mm256_mul_ps(vt5, vs5);
    vs5 = _mm256_sub_ps(vs5, vone);

    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vt0);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vt1);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vt2);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vt3);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vt4);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vt5);

    const __m256 ve0 = _mm256_mul_ps(_mm256_add_ps(vp0, vs0), valpha);
    vx0 = _mm256_mul_ps(vx0, vbeta);
    const __m256 ve1 = _mm256_mul_ps(_mm256_add_ps(vp1, vs1), valpha);
    vx1 = _mm256_mul_ps(vx1, vbeta);
    const __m256 ve2 = _mm256_mul_ps(_mm256_add_ps(vp2, vs2), valpha);
    vx2 = _mm256_mul_ps(vx2, vbeta);
    const __m256 ve3 = _mm256_mul_ps(_mm256_add_ps(vp3, vs3), valpha);
    vx3 = _mm256_mul_ps(vx3, vbeta);
    const __m256 ve4 = _mm256_mul_ps(_mm256_add_ps(vp4, vs4), valpha);
    vx4 = _mm256_mul_ps(vx4, vbeta);
    const __m256 ve5 = _mm256_mul_ps(_mm256_add_ps(vp5, vs5), valpha);
    vx5 = _mm256_mul_ps(vx5, vbeta);

    const __m256 vy0 = _mm256_blendv_ps(vx0, ve0, vx0);
    const __m256 vy1 = _mm256_blendv_ps(vx1, ve1, vx1);
    const __m256 vy2 = _mm256_blendv_ps(vx2, ve2, vx2);
    const __m256 vy3 = _mm256_blendv_ps(vx3, ve3, vx3);
    const __m256 vy4 = _mm256_blendv_ps(vx4, ve4, vx4);
    const __m256 vy5 = _mm256_blendv_ps(vx5, ve5, vx5);

    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    _mm256_storeu_ps(output + 24, vy3);
    _mm256_storeu_ps(output + 32, vy4);
    _mm256_storeu_ps(output + 40, vy5);
    output += 48;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m256 vidx = _mm256_and_ps(vn, vindex_mask);

    const __m128i vidx_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx)), 2);
    const __m128i vidx_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_ll = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx_lh = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx_hl = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      const uint64_t vidx_hh = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_ll)));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lh)));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hl)));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hh)));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_ll >> 32))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lh >> 32))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hl >> 32))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hh >> 32))), 1);
    #else
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_lo))));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 2))));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_hi))));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 2))));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 1))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 3))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 1))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 3))), 1);
    #endif
    const __m128i ven_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 19);
    const __m128i ven_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 19);

    const __m128i vl_lo = _mm_unpacklo_epi64(vl_ll, vl_lh);
    const __m128i vl_hi = _mm_unpacklo_epi64(vl_hl, vl_hh);

    vn = _mm256_sub_ps(vn, vmagic_bias);
    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ven_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ven_hi));

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);
    __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc3, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vt);

    const __m256 ve = _mm256_mul_ps(_mm256_add_ps(vp, vs), valpha);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m256 vidx = _mm256_and_ps(vn, vindex_mask);

    const __m128i vidx_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx)), 2);
    const __m128i vidx_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_ll = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx_lh = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx_hl = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      const uint64_t vidx_hh = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_ll)));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lh)));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hl)));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hh)));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_ll >> 32))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lh >> 32))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hl >> 32))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hh >> 32))), 1);
    #else
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_lo))));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 2))));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_hi))));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 2))));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 1))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 3))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 1))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 3))), 1);
    #endif
    const __m128i ven_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 19);
    const __m128i ven_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 19);

    const __m128i vl_lo = _mm_unpacklo_epi64(vl_ll, vl_lh);
    const __m128i vl_hi = _mm_unpacklo_epi64(vl_hl, vl_hh);

    vn = _mm256_sub_ps(vn, vmagic_bias);
    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ven_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ven_hi));

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);
    __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc3, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vt);

    const __m256 ve = _mm256_mul_ps(_mm256_add_ps(vp, vs), valpha);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

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
