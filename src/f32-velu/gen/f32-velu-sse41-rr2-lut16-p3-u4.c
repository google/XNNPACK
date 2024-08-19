// Auto-generated file. Do not edit!
//   Template: src/f32-velu/sse-rr2-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.154246p+4f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p19f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128i vindex_mask = _mm_set1_epi32(UINT32_C(0xF));
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc3 = _mm_set1_ps(0x1.55561Cp-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.0001ECp-1f);
  const __m128 vone = _mm_set1_ps(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const __m128 vprescale = _mm_set1_ps(params->scalar.prescale);
  const __m128 valpha = _mm_set1_ps(params->scalar.alpha);
  const __m128 vbeta = _mm_set1_ps(params->scalar.beta);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ven = _mm_slli_epi32(_mm_castps_si128(vn), 19);
    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_extract_epi64(vidx, 1);
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi)));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32))), 1);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 2))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 6))), 1);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);
    __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ven));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc3, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    vx = _mm_mul_ps(vx, vbeta);
    const __m128 vy = _mm_blendv_ps(vx, ve, vx);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ven = _mm_slli_epi32(_mm_castps_si128(vn), 19);
    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_extract_epi64(vidx, 1);
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi)));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32))), 1);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 2))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 6))), 1);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);
    __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ven));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc3, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    vx = _mm_mul_ps(vx, vbeta);
    __m128 vy = _mm_blendv_ps(vx, ve, vx);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
