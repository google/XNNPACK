// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/sse-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsign_mask = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.sign_mask);
  const __m128 vsat_cutoff = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.sat_cutoff);
  const __m128 vlog2e = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.log2e);
  const __m128 vmagic_bias = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.magic_bias);
  const __m128i vindex_mask = _mm_load_si128((const __m128i*) params->sse_expm1minus_rr1_lut8_p4h3.index_mask);
  const __m128 vminus_ln2 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.minus_ln2);
  const __m128 vc4 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.c4);
  const __m128 vc3 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.c3);
  const __m128 vc2 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.c2);
  const __m128 vminus_two = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.minus_two);
  const __m128 vminus_one = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.minus_one);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    __m128 vz = _mm_or_ps(vx, vsign_mask);

    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    vz = _mm_max_ps(vsat_cutoff, vz);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 20);

    #if XNN_ARCH_X86_64
      __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      vidx = _mm_unpackhi_epi64(vidx, vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(vidx);
      const __m128i vl0 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
      const __m128i vl1 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)]);
      const __m128i vl_lo = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl2 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi]);
      const __m128i vl3 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)]);
      const __m128i vl_hi = _mm_unpacklo_epi32(vl2, vl3);
    #else
      const __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      const __m128i vl0 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx0]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      const __m128i vl1 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx1]);
      const __m128i vl_lo = _mm_unpacklo_epi32(vl0, vl1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      const __m128i vl2 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx2]);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      const __m128i vl3 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx3]);
      const __m128i vl_hi = _mm_unpacklo_epi32(vl2, vl3);
    #endif
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    vn = _mm_sub_ps(vn, vmagic_bias);

    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc4, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    const __m128 vepo = _mm_sub_ps(vemo, vminus_two);

    __m128 vy = _mm_div_ps(vemo, vepo);


    vy = _mm_xor_ps(vy, vinvsignx);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    __m128 vz = _mm_or_ps(vx, vsign_mask);

    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    vz = _mm_max_ps(vsat_cutoff, vz);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 20);

    #if XNN_ARCH_X86_64
      __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      vidx = _mm_unpackhi_epi64(vidx, vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(vidx);
      const __m128i vl0 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
      const __m128i vl1 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)]);
      const __m128i vl_lo = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl2 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi]);
      const __m128i vl3 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)]);
      const __m128i vl_hi = _mm_unpacklo_epi32(vl2, vl3);
    #else
      const __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      const __m128i vl0 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx0]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      const __m128i vl1 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx1]);
      const __m128i vl_lo = _mm_unpacklo_epi32(vl0, vl1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      const __m128i vl2 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx2]);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      const __m128i vl3 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx3]);
      const __m128i vl_hi = _mm_unpacklo_epi32(vl2, vl3);
    #endif
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    vn = _mm_sub_ps(vn, vmagic_bias);

    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc4, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    const __m128 vepo = _mm_sub_ps(vemo, vminus_two);

    __m128 vy = _mm_div_ps(vemo, vepo);


    vy = _mm_xor_ps(vy, vinvsignx);

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
